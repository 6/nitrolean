//! Weak head normal forms: `whnf_core` (projections, recursors, quotients,
//! let-bound variables; β is done by evaluation) and `whnf` (also δ and the
//! `Nat` operations), as in `type_checker.cpp`.

use num_bigint::BigUint;

use crate::env::{Decl, DeclKind};
use crate::expr::ExprNode;
use crate::level::{LevelId, LevelList};
use crate::name::NameId;

use super::super::ext::{NatOp, SMALL_NAT, SmallNat};
use super::super::tc::{KErr, NAT_MAX_BYTES, R};
use super::val::{Elim, Env, NONE, Sp, V, Val};
use super::{BOTTOM, Nbe};

/// Steps of reduction before a closed computation goes to the closed-term
/// evaluator (`whnf_closed`; 0: at once).
const EVAL_AFTER: u64 = 4096;

/// Whether the closed-term evaluator is asked within an attempt (see
/// `whnf_closed`).
const EVAL_IN_ATTEMPTS: bool = false;

impl<'e> Nbe<'e> {
    /// The value bound to the let-variable `k`, if it is one.
    #[inline]
    fn let_value(&self, k: u32) -> Option<V> {
        if k == BOTTOM {
            return None;
        }
        self.fvars[k as usize].val
    }

    #[inline]
    pub fn whnf_core(&mut self, v: V, cheap_rec: bool, cheap_proj: bool) -> R<V> {
        if self.a.core_normal(v) {
            return Ok(v);
        }
        self.whnf_core_non_normal(v, cheap_rec, cheap_proj)
    }

    /// Core reduction after the original value's normal flag has been checked.
    #[inline]
    fn whnf_core_non_normal(&mut self, mut v: V, cheap_rec: bool, cheap_proj: bool) -> R<V> {
        if self.a.is_thunk(v) {
            v = self.force(v);
            if self.a.core_normal(v) {
                return Ok(v);
            }
        }
        let rec = self.a.r(v);
        if !cheap_rec {
            let c = if cheap_proj {
                rec.cheap_core()
            } else {
                rec.whnf_core
            };
            if c != NONE {
                return Ok(V(c));
            }
        }
        self.whnf_core_slow(v, cheap_rec, cheap_proj)
    }

    fn whnf_core_slow(&mut self, v: V, cheap_rec: bool, cheap_proj: bool) -> R<V> {
        let r = match self.a.get(v) {
            Val::Const(_, _, sp) if sp != Sp::NIL => {
                self.whnf_core_app(v, cheap_rec, cheap_proj)?
            }
            Val::FVar(k, sp) => match self.let_value(k) {
                Some(x) => {
                    let r = self.apply_spine(x, sp);
                    self.whnf_core(r, cheap_rec, cheap_proj)?
                }
                None => {
                    self.a.set_whnf_normal(v);
                    return Ok(v);
                }
            },
            _ => return Ok(v),
        };
        if !cheap_rec {
            let rec = self.a.rm(v);
            if cheap_proj {
                rec.set_cheap_core(r);
            } else {
                rec.whnf_core = r.0;
            }
        }
        Ok(r)
    }

    fn whnf_core_app(&mut self, v: V, cheap_rec: bool, cheap_proj: bool) -> R<V> {
        self.tick()?;
        let Val::Const(c, l, sp) = self.a.get(v) else {
            unreachable!()
        };
        let has_proj = self.a.sp_has_proj(sp);
        let d = self.find(c);
        let is_rec = d.as_ref().is_some_and(|d| match d.kind {
            DeclKind::Recursor { .. } => true,
            DeclKind::Quot { .. } => self.is_quot_recursor(c),
            _ => false,
        });
        if !is_rec {
            if !has_proj {
                if d.is_some() {
                    self.a.set_core_normal(v);
                }
                return Ok(v);
            }
            // Cheap projection leaves this prefix headed by c, so only a
            // constructor prefix can supply the field.
            if cheap_proj && d.is_some_and(|d| !matches!(d.kind, DeclKind::Constructor { .. })) {
                return Ok(v);
            }
        }
        let p = self.a.sp_arg_prefix(sp);
        if let Some(d) = d.filter(|_| is_rec)
            && let Some(r) = self.reduce_recursor(&d, l, sp, p, cheap_rec, cheap_proj)?
        {
            let r = self.apply_sp_suffix(r, sp, p);
            self.whnf_core(r, cheap_rec, cheap_proj)
        } else if p < self.a.sp_len(sp) {
            let Elim::Proj(s, i) = self.a.sp_elim(sp, p) else {
                unreachable!()
            };
            if !cheap_proj
                && !is_rec
                && let Some(d) = d.as_ref()
                && let Some(f) = self.project_definition(d, l, sp, p, (s, i))
            {
                let r = self.apply_sp_suffix(f, sp, p + 1);
                return self.whnf_core(r, cheap_rec, cheap_proj);
            }
            let pre = self.a.sp_prefix(sp, p);
            let x = self.a.mk(Val::Const(c, l, pre));
            if !is_rec && d.is_some() {
                // The prefix has only arguments and its head cannot reduce in core.
                self.a.set_core_normal(x);
            }
            let x = if cheap_proj {
                self.whnf_core(x, cheap_rec, cheap_proj)?
            } else {
                self.whnf(x)?
            };
            match self.reduce_proj_core(x, s, i)? {
                Some(f) => {
                    let r = self.apply_sp_suffix(f, sp, p + 1);
                    self.whnf_core(r, cheap_rec, cheap_proj)
                }
                None => Ok(v),
            }
        } else {
            Ok(v)
        }
    }

    /// A fully applied definition with a constructor body can supply its
    /// field without evaluating the other constructor arguments.
    fn project_definition(
        &mut self,
        d: &Decl,
        ls: LevelList,
        sp: Sp,
        p: u32,
        (s, i): (NameId, u32),
    ) -> Option<V> {
        let mut body = Self::decl_value(d)?;
        if crate::env::is_lazy(body)
            || self.st.list(ls).len() != d.level_params.len()
            || (ls == LevelList::EMPTY
                && (Some(d.name) == self.ext.nat_succ || self.ext.nat_op(d.name).is_some()))
        {
            return None;
        }
        for _ in 0..p {
            let ExprNode::Lam(_, _, _, rest) = self.st.node(body) else {
                return None;
            };
            body = rest;
        }
        let (name, ctor_ls) = self.st.head_const(body)?;
        let ctor = self.find_ref(name)?;
        let DeclKind::Constructor {
            induct,
            num_params,
            num_fields,
            ..
        } = ctor.kind
        else {
            return None;
        };
        if induct != s
            || i >= num_fields
            || (ctor_ls == LevelList::EMPTY
                && (Some(name) == self.ext.nat_succ || self.ext.nat_op(name).is_some()))
        {
            return None;
        }
        let mut field = None;
        let mut len = 0;
        while let ExprNode::App(f, a) = self.st.node(body) {
            if len == num_fields - 1 - i {
                field = Some(a);
            }
            body = f;
            len += 1;
        }
        if len != num_params + num_fields {
            return None;
        }
        let field = field?;
        if let ExprNode::BVar(j) = self.st.node(field)
            && j < p
        {
            return Some(self.a.sp_arg(sp, p - 1 - j));
        }
        let mut env = Env::NIL;
        if self.st.has_loose_bvars(field) {
            for j in 0..p {
                env = self.a.cons(env, self.a.sp_arg(sp, j));
            }
        }
        let ls = self.decl_ls(d, ls);
        Some(self.thunk(field, env, ls))
    }

    fn apply_sp_suffix(&mut self, mut f: V, sp: Sp, start: u32) -> V {
        for i in start..self.a.sp_len(sp) {
            f = self.apply_elim(f, self.a.sp_elim(sp, i));
        }
        f
    }

    fn apply_sp_args(&mut self, f: V, sp: Sp, start: u32, end: u32) -> V {
        let base = self.vargs.len();
        self.a.extend_sp_args(sp, start, end, &mut self.vargs);
        let r = self.apply_n(f, base);
        self.vargs.truncate(base);
        r
    }

    /// Lean's `reduce_proj_core`, on a struct already in whnf.
    pub fn reduce_proj_core(&mut self, mut c: V, s: NameId, i: u32) -> R<Option<V>> {
        if let Val::Str(_) = self.a.get(c) {
            let x = self.string_lit_to_constructor(c)?;
            c = self.whnf(x)?;
        }
        Ok(self.ctor_field(c, s, i))
    }

    fn is_quot_recursor(&self, c: NameId) -> bool {
        self.ext
            .quot
            .as_ref()
            .is_some_and(|q| c == q.lift || c == q.ind)
    }

    fn reduce_recursor(
        &mut self,
        d: &Decl,
        l: LevelList,
        sp: Sp,
        arg_count: u32,
        cheap_rec: bool,
        cheap_proj: bool,
    ) -> R<Option<V>> {
        if matches!(d.kind, DeclKind::Quot { .. }) {
            return self.quot_reduce_rec(d.name, sp, arg_count);
        }
        self.inductive_reduce_rec(d, l, sp, arg_count, cheap_rec, cheap_proj)
    }

    fn quot_reduce_rec(&mut self, f: NameId, sp: Sp, arg_count: u32) -> R<Option<V>> {
        let Some(q) = self.ext.quot.as_ref() else {
            return Ok(None);
        };
        let (mk_pos, arg_pos) = if f == q.lift {
            (5, 3)
        } else if f == q.ind {
            (4, 3)
        } else {
            return Ok(None);
        };
        let qmk = q.mk;
        if arg_count <= mk_pos {
            return Ok(None);
        }
        let mk = self.whnf(self.a.sp_arg(sp, mk_pos))?;
        let Val::Const(n, _, msp) = self.a.get(mk) else {
            return Ok(None);
        };
        if n != qmk || self.a.sp_has_proj(msp) || self.a.sp_len(msp) != 3 {
            return Ok(None);
        }
        let r = self.apply(self.a.sp_arg(sp, arg_pos), self.a.sp_arg(msp, 2));
        Ok(Some(self.apply_sp_args(r, sp, mk_pos + 1, arg_count)))
    }

    pub(super) fn major_induct(&mut self, rec: NameId, d: &Decl) -> Option<NameId> {
        if let Some(&n) = self.major_induct.get(&rec) {
            return Some(n);
        }
        let DeclKind::Recursor {
            num_params,
            num_indices,
            num_motives,
            num_minors,
            ..
        } = d.kind
        else {
            return None;
        };
        let k = num_params + num_motives + num_minors + num_indices;
        let mut t = d.ty;
        for _ in 0..k {
            match self.st.node(t) {
                crate::expr::ExprNode::Pi(_, _, _, b) => t = b,
                _ => return None,
            }
        }
        let crate::expr::ExprNode::Pi(_, _, dom, _) = self.st.node(t) else {
            return None;
        };
        let (n, _) = self.st.head_const(dom)?;
        self.major_induct.insert(rec, n);
        Some(n)
    }

    fn rec_whnf(&mut self, v: V, cheap_rec: bool, cheap_proj: bool) -> R<V> {
        if cheap_rec {
            self.whnf_core(v, cheap_rec, cheap_proj)
        } else {
            self.whnf(v)
        }
    }

    pub fn is_non_rec_structure(&self, n: NameId) -> bool {
        match self.find(n) {
            Some(Decl {
                kind:
                    DeclKind::Inductive {
                        ctors,
                        num_indices,
                        is_rec,
                        ..
                    },
                ..
            }) => ctors.len() == 1 && num_indices == 0 && !is_rec,
            _ => false,
        }
    }

    /// The constant at the head of `v` (forced), with its spine.
    pub fn head_const(&mut self, v: V) -> Option<(NameId, LevelList, Sp)> {
        let v = self.force(v);
        match self.a.get(v) {
            Val::Const(n, l, sp) => Some((n, l, sp)),
            _ => None,
        }
    }

    fn is_constructor_app(&mut self, v: V) -> bool {
        match self.head_const(v) {
            Some((n, _, _)) => matches!(
                self.find(n),
                Some(Decl {
                    kind: DeclKind::Constructor { .. },
                    ..
                })
            ),
            None => false,
        }
    }

    pub fn first_ctor(&self, i: NameId) -> Option<NameId> {
        match self.find(i)?.kind {
            DeclKind::Inductive { ctors, .. } => self.st.name_list(ctors).first().copied(),
            _ => None,
        }
    }

    fn ctor_when_k(&mut self, induct: NameId, nparams: u32, e: V) -> R<V> {
        let t = self.infer_value(e)?;
        let app_type = self.whnf(t)?;
        let Val::Const(i_name, i_lvls, sp) = self.a.get(app_type) else {
            return Ok(e);
        };
        if i_name != induct {
            return Ok(e);
        }
        let Some(c) = self.first_ctor(i_name) else {
            return Ok(e);
        };
        let mut args = Vec::new();
        if !self.a.args(sp, &mut args) || args.len() < nparams as usize {
            return Ok(e);
        }
        let c = self.cnst(c, i_lvls);
        let new_c = self.apply_all(c, &args[..nparams as usize]);
        let new_t = self.infer_value(new_c)?;
        if !self.is_def_eq(app_type, new_t)? {
            return Ok(e);
        }
        Ok(new_c)
    }

    /// `mk params (e.1) … (e.n)` for `e` of the structure type `e_type`.
    pub fn expand_eta_struct(&mut self, e_type: V, e: V) -> R<V> {
        let Val::Const(i_name, i_lvls, sp) = self.a.get(e_type) else {
            return Ok(e);
        };
        let Some(c) = self.first_ctor(i_name) else {
            return Ok(e);
        };
        let DeclKind::Constructor {
            num_params,
            num_fields,
            ..
        } = self.get(c)?.kind
        else {
            return Ok(e);
        };
        let mut args = Vec::new();
        if !self.a.args(sp, &mut args) {
            return Ok(e);
        }
        let np = (num_params as usize).min(args.len());
        let c = self.cnst(c, i_lvls);
        let mut r = self.apply_all(c, &args[..np]);
        for k in 0..num_fields {
            let p = self.proj(e, i_name, k);
            r = self.apply(r, p);
        }
        Ok(r)
    }

    fn is_large_elim(&self, d: &Decl) -> bool {
        let DeclKind::Recursor { all, .. } = d.kind else {
            return false;
        };
        let Some(&first) = self.st.name_list(all).first() else {
            return false;
        };
        match self.find(first) {
            Some(i) => d.level_params.len() != i.level_params.len(),
            None => false,
        }
    }

    fn ctor_when_structure(&mut self, rec: &Decl, induct: NameId, e: V) -> R<V> {
        if !self.is_non_rec_structure(induct) || self.is_constructor_app(e) {
            return Ok(e);
        }
        let ef = self.force(e);
        if matches!(self.a.get(ef), Val::Str(_)) {
            return Ok(e);
        }
        let t = self.infer_value(e)?;
        let e_type = self.whnf(t)?;
        match self.head_const(e_type) {
            Some((n, _, _)) if n == induct => {}
            _ => return Ok(e),
        }
        // Divergence (corpus/divergences.md), as in `tc.rs`.
        if !self.is_large_elim(rec) {
            return Ok(e);
        }
        self.expand_eta_struct(e_type, e)
    }

    fn inductive_reduce_rec(
        &mut self,
        d: &Decl,
        rec_lvls: LevelList,
        sp: Sp,
        arg_count: u32,
        cheap_rec: bool,
        cheap_proj: bool,
    ) -> R<Option<V>> {
        let DeclKind::Recursor {
            num_params,
            num_indices,
            num_motives,
            num_minors,
            rules,
            k,
            ..
        } = d.kind
        else {
            return Ok(None);
        };
        let major_idx = num_params + num_motives + num_minors + num_indices;
        if major_idx >= arg_count {
            return Ok(None);
        }
        let Some(induct) = self.major_induct(d.name, d) else {
            return Ok(None);
        };
        let mut major = self.a.sp_arg(sp, major_idx);
        if k {
            major = self.ctor_when_k(induct, num_params, major)?;
        }
        major = self.ctor_when_structure(d, induct, major)?;
        major = self.rec_whnf(major, cheap_rec, cheap_proj)?;
        match self.a.get(major) {
            Val::Nat(..) | Val::NatBig(_) => {
                major = self.nat_lit_to_constructor(major)?;
            }
            Val::Str(_) => {
                let x = self.string_lit_to_constructor(major)?;
                major = self.whnf(x)?;
            }
            _ => {}
        }
        let Val::Const(ctor, _, msp) = self.a.get(major) else {
            return Ok(None);
        };
        let Some(rule) = self
            .st
            .rec_rules(rules)
            .iter()
            .find(|r| r.ctor == ctor)
            .copied()
        else {
            return Ok(None);
        };
        let mlen = self.a.sp_len(msp);
        if self.a.sp_has_proj(msp) || rule.nfields > mlen {
            return Ok(None);
        }
        if self.st.list(rec_lvls).len() != d.level_params.len() {
            return Ok(None);
        }
        let ls = self.decl_ls(d, rec_lvls);
        let mut rhs = self.eval(rule.rhs, super::val::Env::NIL, ls);
        let pmm = num_params + num_motives + num_minors;
        rhs = self.apply_sp_args(rhs, sp, 0, pmm);
        rhs = self.apply_sp_args(rhs, msp, mlen - rule.nfields, mlen);
        rhs = self.apply_sp_args(rhs, sp, major_idx + 1, arg_count);
        Ok(Some(rhs))
    }

    /// `n` as `Nat.zero` or `Nat.succ (n - 1)`.
    pub fn nat_lit_to_constructor(&mut self, v: V) -> R<V> {
        if SMALL_NAT
            && let Val::Nat(lo, hi) = self.a.get(v)
            && let (Some(zero), Some(succ)) = (self.ext.nat_zero, self.ext.nat_succ)
        {
            let n = (hi as u64) << 32 | lo as u64;
            if n == 0 {
                return Ok(self.cnst(zero, LevelList::EMPTY));
            }
            self.st
                .charge_nat_unfold(u64::from(u64::BITS - n.leading_zeros()));
            let p = self.nat_u64(n - 1);
            let s = self.cnst(succ, LevelList::EMPTY);
            return Ok(self.apply(s, p));
        }
        let Some(n) = self.nat_value(v) else {
            unreachable!()
        };
        let (Some(zero), Some(succ)) = (self.ext.nat_zero, self.ext.nat_succ) else {
            return Err(KErr::new(
                "a `Nat` literal needs the prelude's `Nat`".into(),
            ));
        };
        if n == BigUint::ZERO {
            return Ok(self.cnst(zero, LevelList::EMPTY));
        }
        self.st.charge_nat_unfold(n.bits());
        let p = self.nat(n - 1u32);
        let s = self.cnst(succ, LevelList::EMPTY);
        Ok(self.apply(s, p))
    }

    /// `String.ofList [Char.ofNat c₁, …]` for a string literal.
    pub fn string_lit_to_constructor(&mut self, v: V) -> R<V> {
        let Val::Str(s) = self.a.get(v) else {
            unreachable!()
        };
        let Some(x) = self.ext.string() else {
            return Err(KErr::new(
                "a `String` literal needs the prelude's `String`".into(),
            ));
        };
        let x = *x;
        let chars: Vec<u32> = self.st.str(s).chars().map(|c| c as u32).collect();
        let zero = LevelList::EMPTY;
        let char_ty = self.cnst(x.char, zero);
        let zero_list = self.st.mk_list(&[LevelId::ZERO]);
        let nil = self.cnst(x.list_nil, zero_list);
        let cons = self.cnst(x.list_cons, zero_list);
        let nil = self.apply(nil, char_ty);
        let cons = self.apply(cons, char_ty);
        let of_nat = self.cnst(x.char_of_nat, zero);
        let mut r = nil;
        for &c in chars.iter().rev() {
            let n = self.nat_u64(c as u64);
            let ch = self.apply(of_nat, n);
            let a = self.apply(cons, ch);
            r = self.apply(a, r);
        }
        let of_list = self.cnst(x.of_list, zero);
        Ok(self.apply(of_list, r))
    }

    /// The definition at the head of `v`, if `v` can be unfolded.
    pub fn is_delta(&mut self, v: V) -> Option<Decl> {
        let v = self.force(v);
        self.delta_definition_forced(v).copied()
    }

    /// The definition at the head of a value already forced by core reduction.
    pub(super) fn delta_definition_forced(&self, v: V) -> Option<&'e Decl> {
        debug_assert!(!self.a.is_thunk(v));
        self.delta_head(v).map(|(d, _, _)| d)
    }

    /// The definition, levels and spine of a forced delta redex.
    fn delta_head(&self, v: V) -> Option<(&'e Decl, LevelList, Sp)> {
        let Val::Const(n, ls, sp) = self.a.get(v) else {
            return None;
        };
        if self.a.sp_has_proj(sp) {
            return None;
        }
        self.delta_decl(n, ls).map(|d| (d, ls, sp))
    }

    fn delta_decl(&self, n: NameId, ls: LevelList) -> Option<&'e Decl> {
        let d = self.find_ref(n)?;
        Self::decl_value(d)?;
        (self.st.list(ls).len() == d.level_params.len()).then_some(d)
    }

    pub fn unfold_definition(&mut self, v: V) -> Option<V> {
        let v = self.force(v);
        let m = self.a.r(v).delta;
        if m != NONE {
            return Some(V(m));
        }
        let (d, ls, sp) = self.delta_head(v)?;
        self.unfold_delta(v, d, ls, sp)
    }

    /// Take the chosen lazy delta step on a forced value with validated metadata.
    pub(super) fn unfold_cheap(&mut self, v: V, d: &Decl) -> R<V> {
        debug_assert!(!self.a.is_thunk(v));
        let m = self.a.r(v).delta;
        let r = if m != NONE {
            V(m)
        } else {
            let Val::Const(_, ls, sp) = self.a.get(v) else {
                unreachable!()
            };
            if super::cases::CASES_DIRECT
                && let Some(branch) = self.cases_direct(d, ls, sp)?
            {
                branch
            } else {
                self.unfold_delta(v, d, ls, sp).expect("delta")
            }
        };
        self.whnf_core(r, false, true)
    }

    fn unfold_delta(&mut self, v: V, d: &Decl, ls: LevelList, sp: Sp) -> Option<V> {
        let value = self.unfold_value(d)?;
        let r = if sp == Sp::NIL {
            self.const_value(d, value, ls)
        } else {
            let len = self.a.sp_len(sp);
            let mut body = value;
            let mut applied = 0;
            while applied < len
                && let ExprNode::Lam(_, _, _, rest) = self.st.node(body)
            {
                body = rest;
                applied += 1;
            }
            if applied == 0 {
                let f = self.const_value(d, value, ls);
                self.apply_sp_args(f, sp, 0, len)
            } else {
                // The skipped binders are the only bindings in scope.
                // Build the environment of the body from the arguments it can read.
                let data = self.st.data(body);
                let mut env = Env::NIL;
                for j in applied - data.loose_bvar_range().min(applied)..applied {
                    let arg = if data.may_use_bvar(applied - 1 - j) {
                        self.a.sp_arg(sp, j)
                    } else {
                        self.hole()
                    };
                    env = self.a.cons(env, arg);
                }
                let ls = self.decl_ls(d, ls);
                let f = self.eval(body, env, ls);
                if applied == len {
                    f
                } else {
                    self.apply_sp_args(f, sp, applied, len)
                }
            }
        };
        self.a.rm(v).delta = r.0;
        Some(r)
    }

    /// The value of a literal below 2^64 or of `Nat.zero`.
    fn nat_small_ext(&self, v: V) -> Option<u64> {
        match self.a.get(v) {
            Val::Nat(lo, hi) => Some((hi as u64) << 32 | lo as u64),
            Val::Const(n, ls, Sp::NIL)
                if Some(n) == self.ext.nat_zero && ls == LevelList::EMPTY =>
            {
                Some(0)
            }
            _ => None,
        }
    }

    /// Whether `v` is a literal or `Nat.zero`.
    fn is_nat_lit_ext(&self, v: V) -> bool {
        match self.a.get(v) {
            Val::Nat(..) | Val::NatBig(_) => true,
            Val::Const(n, ls, Sp::NIL) => Some(n) == self.ext.nat_zero && ls == LevelList::EMPTY,
            _ => false,
        }
    }

    /// The value of a literal or of `Nat.zero`.
    fn nat_lit_ext(&self, v: V) -> Option<BigUint> {
        match self.a.get(v) {
            Val::Const(n, ls, Sp::NIL)
                if Some(n) == self.ext.nat_zero && ls == LevelList::EMPTY =>
            {
                Some(BigUint::ZERO)
            }
            _ => self.nat_value(v),
        }
    }

    fn nat_size_ok(v: &BigUint) -> R<()> {
        if v.bits().div_ceil(8) > NAT_MAX_BYTES {
            return Err(KErr::new(
                "the kernel refused a `Nat` numeral because its size exceeds the maximum".into(),
            ));
        }
        Ok(())
    }

    fn mk_bool(&mut self, b: bool) -> R<V> {
        let n = if b {
            self.ext.bool_true
        } else {
            self.ext.bool_false
        };
        match n {
            Some(n) => Ok(self.cnst(n, LevelList::EMPTY)),
            None => Err(KErr::new("`Bool` is missing".into())),
        }
    }

    pub fn reduce_nat(&mut self, v: V) -> R<Option<V>> {
        if self.ext.nat.is_none() {
            return Ok(None);
        }
        let v = self.force(v);
        let Val::Const(n, ls, sp) = self.a.get(v) else {
            return Ok(None);
        };
        if ls != LevelList::EMPTY || self.a.sp_has_proj(sp) {
            return Ok(None);
        }
        self.reduce_nat_app(n, sp)
    }

    fn reduce_nat_app(&mut self, n: NameId, sp: Sp) -> R<Option<V>> {
        match self.a.sp_len(sp) {
            1 if Some(n) == self.ext.nat_succ => {
                let Elim::Arg(a) = self.a.sp_elim(sp, 0) else {
                    unreachable!()
                };
                let a = self.whnf(a)?;
                if SMALL_NAT
                    && let Some(x) = self.nat_small_ext(a)
                    && let Some(r) = x.checked_add(1)
                {
                    return Ok(Some(self.nat_u64(r)));
                }
                let Some(x) = self.nat_lit_ext(a) else {
                    return Ok(None);
                };
                let x = x + 1u32;
                Self::nat_size_ok(&x)?;
                Ok(Some(self.nat(x)))
            }
            2 => {
                let Some(op) = self.ext.nat_op(n) else {
                    return Ok(None);
                };
                let e_r = self.a.sp_elim(sp, 1);
                let e_l = self.a.sp_elim(sp, 0);
                let (Elim::Arg(a_l), Elim::Arg(a_r)) = (e_l, e_r) else {
                    unreachable!()
                };
                let x = self.whnf(a_l)?;
                if !self.is_nat_lit_ext(x) {
                    return Ok(None);
                }
                let y = self.whnf(a_r)?;
                if SMALL_NAT
                    && let (Some(a), Some(b)) = (self.nat_small_ext(x), self.nat_small_ext(y))
                    && let Some(r) = op.small(a, b)
                {
                    return match r {
                        SmallNat::Nat(n) => Ok(Some(self.nat_u64(n))),
                        SmallNat::Bool(b) => self.mk_bool(b).map(Some),
                    };
                }
                let Some(x) = self.nat_lit_ext(x) else {
                    return Ok(None);
                };
                let Some(y) = self.nat_lit_ext(y) else {
                    return Ok(None);
                };
                self.nat_op(op, x, y).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn nat_op(&mut self, op: NatOp, x: BigUint, y: BigUint) -> R<V> {
        let r = match op {
            NatOp::Add => x + y,
            NatOp::Sub => {
                if x >= y {
                    x - y
                } else {
                    BigUint::ZERO
                }
            }
            NatOp::Mul => x * y,
            NatOp::Div => {
                if y == BigUint::ZERO {
                    BigUint::ZERO
                } else {
                    x / y
                }
            }
            NatOp::Mod => {
                if y == BigUint::ZERO {
                    x
                } else {
                    x % y
                }
            }
            NatOp::Gcd => {
                let (mut a, mut b) = (x, y);
                while b != BigUint::ZERO {
                    let r = &a % &b;
                    a = b;
                    b = r;
                }
                a
            }
            NatOp::Beq => return self.mk_bool(x == y),
            NatOp::Ble => return self.mk_bool(x <= y),
            NatOp::Land => x & y,
            NatOp::Lor => x | y,
            NatOp::Xor => x ^ y,
            NatOp::Pow => {
                let Ok(k) = u32::try_from(&y) else {
                    return Err(KErr::new(
                        "the kernel refused to evaluate `Nat.pow` because its second argument does not fit in a 32-bit unsigned integer".into(),
                    ));
                };
                if x > BigUint::from(1u32)
                    && k != 0
                    && x.bits().div_ceil(8) > NAT_MAX_BYTES / k as u64
                {
                    return Err(KErr::new(
                        "the kernel refused to evaluate `Nat.pow` because the result would exceed the maximum numeral size".into(),
                    ));
                }
                x.pow(k)
            }
            NatOp::ShiftLeft => {
                if x == BigUint::ZERO {
                    BigUint::ZERO
                } else {
                    let Ok(k) = u32::try_from(&y) else {
                        return Err(KErr::new(
                            "the kernel refused to evaluate `Nat.shiftLeft` because its second argument does not fit in a 32-bit unsigned integer".into(),
                        ));
                    };
                    if x.bits().div_ceil(8) + k as u64 / 8 + 1 > NAT_MAX_BYTES {
                        return Err(KErr::new(
                            "the kernel refused a `Nat` numeral because its size exceeds the maximum".into(),
                        ));
                    }
                    x << k
                }
            }
            NatOp::ShiftRight => match u64::try_from(&y) {
                Ok(k) if k < x.bits() => x >> k,
                _ => BigUint::ZERO,
            },
        };
        if matches!(op, NatOp::Add | NatOp::Sub | NatOp::Mul) {
            Self::nat_size_ok(&r)?;
        }
        Ok(self.nat(r))
    }

    /// `f` with the closed-term evaluator not asked (an unwind out of it
    /// leaves the flag set: `begin` and `reset` clear it).
    pub(super) fn probe<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.eval_probe = true;
        let r = f(self);
        self.eval_probe = false;
        r
    }

    /// The weak head normal form of `t`, a closed definition applied, if
    /// it is worth asking the closed-term evaluator: past `EVAL_AFTER`
    /// steps of reduction here (most computations are a few steps, where
    /// converting to terms and back costs more than it saves).
    fn whnf_closed(&mut self, t: V) -> R<Option<V>> {
        if self.eval_probe || !self.closed_eval || self.a.has_fvar(t) {
            return Ok(None);
        }
        if self.stop != u64::MAX {
            // Within an attempt (a probe of a few steps, which gives up
            // past its budget), a computation worth the evaluator does not
            // finish anyway: it is redone, and evaluated, outside.
            return Ok(if EVAL_IN_ATTEMPTS {
                self.closed_whnf(t)
            } else {
                None
            });
        }
        if EVAL_AFTER == 0 {
            return Ok(self.closed_whnf(t));
        }
        let r = self.probe(|nb| nb.attempt(EVAL_AFTER, |nb| nb.whnf_from(t, false)));
        match r? {
            Some(n) => Ok(Some(n)),
            None => Ok(self.closed_whnf(t)),
        }
    }

    #[inline]
    pub fn whnf(&mut self, v: V) -> R<V> {
        if self.a.whnf_normal(v) {
            return Ok(v);
        }
        self.whnf_slow(v)
    }

    fn whnf_slow(&mut self, v: V) -> R<V> {
        let v = self.force(v);
        if self.a.whnf_normal(v) {
            return Ok(v);
        }
        let c = self.a.r(v).whnf;
        if c != NONE {
            return Ok(V(c));
        }
        if let Val::FVar(k, _) = self.a.get(v)
            && self.let_value(k).is_none()
        {
            self.a.set_whnf_normal(v);
            return Ok(v);
        }
        if self.stable_queries.is_some() && !self.eval_probe && !self.a.has_fvar(v) {
            return self.whnf_stable(v);
        }
        self.whnf_from(v, true)
    }

    fn whnf_stable(&mut self, v: V) -> R<V> {
        // A cached reduct may expose another reduction in the larger
        // environment. Always continue normalizing it there.
        if let Some(reduct) = self.stable_reduct(v) {
            let n = self.whnf_from(reduct, true)?;
            self.a.rm(v).whnf = n.0;
            return Ok(n);
        }
        let start = self.ticks;
        let n = self.whnf_from(v, true)?;
        if n != v {
            self.record_stable_reduct(v, n, start);
        }
        Ok(n)
    }

    /// `whnf` of `v`, by the closed-term evaluator too if `eval`.
    fn whnf_from(&mut self, v: V, eval: bool) -> R<V> {
        let mut t = v;
        let mut first = eval;
        let n = loop {
            self.tick()?;
            if self.a.whnf_normal(t) {
                break t;
            }
            let w = self.a.r(t).whnf;
            if w != NONE {
                break V(w);
            }
            let t_l = if self.a.core_normal(t) {
                t
            } else {
                let r = self.whnf_core_non_normal(t, false, false)?;
                if self.a.whnf_normal(r) {
                    break r;
                }
                r
            };
            let Val::Const(name, ls, sp) = self.a.get(t_l) else {
                break t_l;
            };
            if self.a.sp_has_proj(sp) {
                break t_l;
            }
            if ls == LevelList::EMPTY
                && self.ext.nat.is_some()
                && let Some(n) = self.reduce_nat_app(name, sp)?
            {
                break n;
            }
            let Some(d) = self.delta_decl(name, ls) else {
                break t_l;
            };
            if first {
                first = false;
                if let Some(n) = self.whnf_closed(t_l)? {
                    self.a.rm(n).whnf = n.0;
                    break n;
                }
            }
            if super::cases::CASES_DIRECT
                && let Some(n) = self.cases_direct(d, ls, sp)?
            {
                t = n;
                continue;
            }
            let m = self.a.r(t_l).delta;
            if m != NONE {
                t = V(m);
            } else if let Some(n) = self.unfold_delta(t_l, d, ls, sp) {
                t = n;
            } else {
                break t_l;
            }
        };
        self.a.rm(v).whnf = n.0;
        Ok(n)
    }
}
