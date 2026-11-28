//! Memos keyed by a term, its universe substitution and the entries of its
//! environment it reads: terms met again in environments that agree where
//! they look find what was found, with no environment cut to build a key.

use hashbrown::HashTable;

use super::val::{Env, Ls, V, spread};
use super::{Nbe, val};
use crate::expr::ExprId;
use crate::hash::{Table, fold32, hash_front, hash2};

/// Whether keys hold the id of the values read (`read_tuple`), not the
/// values.
const READ_TUPLES: bool = true;

/// Whether a key of a term that reads no value of its range (all holes)
/// takes a fixed id for the values read, with no tuple looked up.
const FAST_KEYS: bool = true;

/// The id of no values read (`FAST_KEYS`).
const EMPTY_TUPLE: u32 = val::NONE - 1;

/// The bit of a tuple id that is an environment's id (`read_tuple`).
const WHOLE_ENV: u32 = 1 << 31;

/// Slots of `Nbe::tuple_slots`.
pub(super) const TUPLE_SLOTS: usize = 1 << 13;

/// Interned tuples store the range, two hole words, and the values read.
#[derive(Default)]
pub(super) struct TupleMemo {
    table: Table,
    words: Vec<u32>,
}

impl TupleMemo {
    pub(super) fn heap_bytes(&self) -> usize {
        self.table.capacity() * size_of::<u64>() + self.words.capacity() * size_of::<u32>()
    }

    pub(super) fn clear(&mut self) {
        if self.table.capacity() > 1 << 16 {
            *self = Self::default();
        } else {
            self.table.clear();
            self.words.clear();
        }
    }
}

/// A memo of `WORDS` words per key (see the module).
pub struct EnvMemo<const WORDS: usize> {
    table: HashTable<u32>,
    /// Records: hash, term, substitution, range, holes (two words), number
    /// of values read, those values, then the `WORDS` words.
    keys: Vec<u32>,
}

impl<const WORDS: usize> Default for EnvMemo<WORDS> {
    fn default() -> Self {
        EnvMemo {
            table: HashTable::new(),
            keys: Vec::new(),
        }
    }
}

impl<const WORDS: usize> EnvMemo<WORDS> {
    /// Bytes held, roughly.
    pub fn heap_bytes(&self) -> usize {
        self.table.capacity() * 5 + self.keys.capacity() * 4
    }

    pub fn clear(&mut self) {
        if self.table.capacity() > 1 << 16 {
            *self = Self::default();
        } else {
            self.table.clear();
            self.keys.clear();
        }
    }

    /// The words at `i` (from `find_or_add`).
    #[inline]
    pub fn get(&self, i: usize) -> &[u32] {
        val::at_slice(&self.keys, i, i + WORDS)
    }

    #[inline]
    pub fn set(&mut self, i: usize, w: usize, v: u32) {
        self.keys[i + w] = v;
    }
}

/// Inference keys use a tuple id and hold the inferred type and check scope.
#[derive(Default)]
pub(super) struct InferMemo {
    table: Table,
    entries: Vec<InferEntry>,
}

struct InferEntry {
    key: [u32; 3],
    /// The type and its check scope: zero for unchecked, or `u32::MAX`
    /// when the check is independent of the universe parameter list.
    result: [u32; 2],
}

impl InferMemo {
    /// Bytes held: the table's 8-byte slots and the entries.
    pub fn heap_bytes(&self) -> usize {
        self.table.capacity() * size_of::<u64>() + self.entries.capacity() * size_of::<InferEntry>()
    }

    #[inline]
    pub(super) fn find_or_add(&mut self, e: ExprId, ls: Ls, tuple: u32) -> usize {
        let key = [e.0, ls.0, tuple];
        let hash = fold32(hash_front(e.0 as u64 | (ls.0 as u64) << 32, tuple as u64));
        let slot = match self
            .table
            .find_or_slot(hash, |i| val::at(&self.entries, i as usize).key == key)
        {
            Ok(i) => return i as usize,
            Err(slot) => slot,
        };
        let i = self.entries.len();
        assert!(
            i < val::NONE as usize,
            "nitrolean: resource limit: memo too large"
        );
        self.entries.push(InferEntry {
            key,
            result: [val::NONE, 0],
        });
        self.table.insert_at(slot, hash, i as u32);
        i
    }

    #[inline]
    pub(super) fn get(&self, i: usize) -> &[u32; 2] {
        &val::at(&self.entries, i).result
    }

    #[inline]
    pub(super) fn set(&mut self, i: usize, w: usize, v: u32) {
        self.entries[i].result[w] = v;
    }

    pub(super) fn clear(&mut self) {
        if self.table.capacity() > 1 << 17 {
            *self = Self::default();
        } else {
            self.table.clear();
            self.entries.clear();
        }
    }
}

/// A key being looked up: its hash and fixed words; the values it reads
/// are on `Nbe::vargs` from `base`.
pub struct Key {
    h: u32,
    head: [u32; 6],
    base: usize,
}

impl Key {
    /// Hash, term, substitution and the id of the values read, if the key
    /// holds the id (`READ_TUPLES`).
    #[inline]
    pub fn by_tuple(&self) -> Option<[u32; 4]> {
        (self.head[2] == val::NONE).then_some([self.h, self.head[0], self.head[1], self.head[3]])
    }

    /// Drop the key, taking the values it read off `vargs`.
    #[inline]
    pub fn drop_in(self, vargs: &mut Vec<V>) {
        vargs.truncate(self.base);
    }
}

impl Nbe<'_> {
    /// The key of `e` in `env` under `ls`, reading the entries below its
    /// range but those `holes` sets (from bit 64 on, all are read); `None`
    /// past `max` entries in range. The values read are left on `vargs`
    /// (give the key to `memo_find_or_add`, which takes them off).
    #[inline]
    pub(super) fn memo_key(
        &mut self,
        e: ExprId,
        env: Env,
        ls: Ls,
        holes: u64,
        max: u32,
    ) -> Option<Key> {
        let d = self.st.data(e);
        let n = d.loose_bvar_range().min(self.a.env_len(env));
        if n > max {
            return None;
        }
        let ls2 = if d.has_level_param() { ls } else { Ls::ID };
        let holes = if n >= 64 {
            holes
        } else {
            holes & ((1u64 << n) - 1)
        };
        if READ_TUPLES {
            let t = self.memo_tuple(env, d.loose_bvar_range(), n, holes);
            let h = hash_front(e.0 as u64 | (ls2.0 as u64) << 32, t as u64);
            return Some(Key {
                h: (h >> 32) as u32,
                head: [e.0, ls2.0, val::NONE, t, 0, 0],
                base: self.vargs.len(),
            });
        }
        let base = self.vargs.len();
        let mut cur = env;
        let mut h = hash2(
            e.0 as u64 | (ls2.0 as u64) << 32,
            n as u64 ^ holes.rotate_left(17),
        );
        for j in 0..n {
            if j >= 64 || holes >> j & 1 == 0 {
                let v = self.a.env_head(cur);
                h = (h.rotate_left(5) ^ v.0 as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                self.vargs.push(v);
            }
            cur = self.a.env_tail(cur);
        }
        let cnt = (self.vargs.len() - base) as u32;
        Some(Key {
            h: (h >> 32) as u32,
            head: [e.0, ls2.0, n, holes as u32, (holes >> 32) as u32, cnt],
            base,
        })
    }

    /// The tuple id for a term's environment, with `holes` masked to `n`.
    #[inline]
    pub(super) fn memo_tuple(&mut self, env: Env, range: u32, n: u32, holes: u64) -> u32 {
        if FAST_KEYS && n == range && n < 64 && n == holes.count_ones() {
            // With the whole range present, every empty list is the same.
            return EMPTY_TUPLE;
        }
        self.read_tuple(env, n, holes)
    }

    /// The id of the values of the first `n` entries of `env` but those
    /// `holes` sets (from bit 64 on, all): equal values, equal ids.
    fn read_tuple(&mut self, env: Env, n: u32, holes: u64) -> u32 {
        // All of a whole environment: environments are hash-consed, so its
        // id names its values (with the top bit, which no record id has).
        if n < 64 && holes & ((1u64 << n) - 1) == 0 && n == self.a.env_len(env) {
            return env.0 | WHOLE_ENV;
        }
        let si = (crate::hash::hash_front(env.0 as u64 | (n as u64) << 32, holes) >> 50) as usize
            & (TUPLE_SLOTS - 1);
        let s = self.tuple_slots[si];
        if s[0] == env.0 && s[1] == n && s[2] == holes as u32 && s[3] == (holes >> 32) as u32 {
            return s[4];
        }
        let t = self.read_tuple_in_place(env, n, holes);
        self.tuple_slots[si] = [env.0, n, holes as u32, (holes >> 32) as u32, t];
        t
    }

    /// `read_tuple` past its front: the record is written at the end of
    /// `tuples` as it is read, and dropped if it is there already. Out of
    /// line, so that a call met in the front costs no frame for it.
    #[inline(never)]
    fn read_tuple_in_place(&mut self, env: Env, n: u32, holes: u64) -> u32 {
        let TupleMemo { table, words } = &mut self.tuples;
        let base = words.len();
        let head = [n, holes as u32, (holes >> 32) as u32];
        let mut cur = env;
        let mut mask = holes;
        let mut h = hash_front(n as u64, holes);
        let skipped = if n >= 64 {
            holes.count_ones()
        } else {
            (holes & ((1u64 << n) - 1)).count_ones()
        };
        // Segment by segment, innermost first.
        let a = &self.a;
        words.reserve(head.len() + (n - skipped) as usize);
        words.extend(head);
        let mut left = n;
        while left > 0 {
            let (seg, base) = a.env_seg(cur);
            let take = (seg.len() as u32).min(left);
            for &v in seg[seg.len() - take as usize..].iter().rev() {
                if mask & 1 == 0 {
                    h = (h.rotate_left(5) ^ v.0 as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                    words.push(v.0);
                }
                mask >>= 1;
            }
            left -= take;
            cur = base;
        }
        let end = words.len();
        let len = end - base;
        let h = fold32(h);
        let slot = match table.find_or_slot(h, |i| {
            let i = i as usize;
            // Equal ranges and hole masks give equal numbers of values.
            // Word by word: a call to compare memory costs more than the
            // loop for records this short.
            (0..len).all(|j| *val::at(words, i + j) == *val::at(words, base + j))
        }) {
            Ok(i) => {
                words.truncate(base);
                return i;
            }
            Err(slot) => slot,
        };
        assert!(
            end < WHOLE_ENV as usize,
            "nitrolean: resource limit: memo too large"
        );
        let id = base as u32;
        table.insert_at(slot, h, id);
        id
    }

    /// The words of `k` in `memo` (by index, for `EnvMemo::get` and `set`),
    /// and whether they were there already; new words are `init`.
    pub(super) fn memo_find_or_add<const WORDS: usize>(
        vargs: &mut Vec<V>,
        memo: &mut EnvMemo<WORDS>,
        k: Key,
        init: [u32; WORDS],
    ) -> (usize, bool) {
        let vals = &vargs[k.base..];
        let EnvMemo { table, keys } = memo;
        let cnt = vals.len();
        let eq = |&i: &u32| {
            let i = i as usize;
            *val::at(keys, i) == k.h
                && (0..6).all(|j| *val::at(keys, i + 1 + j) == k.head[j])
                && val::at_slice(keys, i + 7, i + 7 + cnt)
                    .iter()
                    .zip(vals)
                    .all(|(&a, b)| a == b.0)
        };
        let r = match table.find(spread(k.h), eq) {
            Some(&i) => (i as usize + 7 + cnt, true),
            None => {
                let i = keys.len();
                assert!(
                    i + 7 + cnt + WORDS < val::NONE as usize,
                    "nitrolean: resource limit: memo too large"
                );
                keys.push(k.h);
                keys.extend_from_slice(&k.head);
                keys.extend(vals.iter().map(|v| v.0));
                keys.extend_from_slice(&init);
                let keys = &*keys;
                table.insert_unique(spread(k.h), i as u32, |&i| spread(keys[i as usize]));
                (i + 7 + cnt, false)
            }
        };
        vargs.truncate(k.base);
        r
    }
}
