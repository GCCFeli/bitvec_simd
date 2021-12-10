//! ## BitVec implemented with [wide](https://github.com/Lokathor/wide)
//!
//! BitVec represents numbers by the position of bits. For example, for the set $\{1,3,5\}$, we
//! can represent it by a just a byte `010101000` -- the most left (high) bit represent if `0`
//! exits in this set or not, the second bit represent `1` ...
//!
//! BitVec is usually used in the algorithm which requires many set intersection/union operations,
//! such like graph mining, formal concept analysis. Set operations in bitvec can be implemented
//! with simple and/or/xor operations so it is much faster than "normal" version of `HashSet`.
//!
//! Furthermore, as SIMD introduces the ability for handling multiple data with a single instruction,
//! set operations can be even faster with SIMD enabled.
//!
//! However, implementation with SIMD in Rust is not really an easy task -- now only low-level API
//! is provided through [core::arch](https://doc.rust-lang.org/core/arch/index.html). It requires
//! many `cfg(target_arch)`s (i.e. different implement on different arch) and
//! assembly-like unsafe function calls.
//!
//! Wide provided a much better API for users. With this crate, you can just treat SIMD
//! operations as an operation on slices. Wide wraps all the low-level details for you -- no
//! arch-specified code, no unsafe, just do what you've done on normal integer/floats.
//!
//! This crate uses Wide to implement a basic bitvec.
//!
//! ### Usage
//!
//! ```rust
//! use bitvec_simd::BitVec;
//!
//! let mut bitvec = BitVec::ones(1_000); //create a set containing 0 ..= 999
//! bitvec.set(1_999, true); // add 1999 to the set, bitvec will be automatically expanded
//! bitvec.set(500, false); // delete 500 from the set
//! // now the set contains: 0 ..=499, 501..=1999
//! assert_eq!(bitvec.get(500), Some(false));
//! assert_eq!(bitvec.get(5_000), None);
//! // When try to get number larger than current bitvec, it will return `None`.
//! // of course if you don't care, you can just do:
//! assert_eq!(bitvec.get(5_000).unwrap_or(false), false);
//!
//! let bitvec2 = BitVec::zeros(2000); // create a set containing 0 ..=1999
//!
//! let bitvec3 = bitvec.and_cloned(&bitvec2);
//! // and/or/xor/not operation is provided.
//! // these APIs usually have 2 version:
//! // `.and` consume the inputs and `.and_clone()` accepts reference and will do clone on inputs.
//! let bitvec4 = bitvec & bitvec2;
//! // ofcourse you can just use bit-and operator on bitvecs, it will also consumes the inputs.
//! assert_eq!(bitvec3, bitvec4);
//! // A bitvec can also be constructed from a collection of bool, or a colluction of integer:
//! let bitvec: BitVec = (0 .. 10).map(|x| x%2 == 0).into();
//! let bitvec2: BitVec = (0 .. 10).map(|x| x%3 == 0).into();
//! let bitvec3 = BitVec::from_bool_iterator((0..10).map(|x| x%6 == 0));
//! assert_eq!(bitvec & bitvec2, bitvec3)
//! ```
//!
//! ## Performance
//!
//! run `cargo bench` to see the benchmarks on your device.

use std::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor, Index, Not},
};

use wide::*;

const BIT_WIDTH: usize = 256;

// BitContainer is the basic building block for internal storage
// BitVec is expected to be aligned properly
type BitContainer = u64x4;

/// Representation of a BitVec
///
/// see the module's document for examples and details.
///
#[derive(Debug, Clone)]
pub struct BitVec {
    // internal representation of bitvec
    storage: Vec<BitContainer>,
    // actual number of bits exists in storage
    nbits: usize,
}

/// Proc macro can not export BitVec
/// macro_rules! can not cancot ident
/// so we use name, name_2 for function names
macro_rules! impl_operation {
    ($name:ident, $name_2:ident, $op:tt) => {
        pub fn $name(self, other: Self) -> Self {
        assert_eq!(self.nbits, other.nbits);
        let storage = self
            .storage
            .into_iter()
            .zip(other.storage.into_iter())
            .map(|(a, b)| a $op b)
            .collect();
        Self {
            storage,
            nbits: self.nbits,
        }
        }
        pub fn $name_2(&self, other: &Self) -> Self {
        assert_eq!(self.nbits, other.nbits);
        let storage = self
            .storage
            .iter()
            .cloned()
            .zip(other.storage.iter().cloned())
            .map(|(a, b)| a $op b)
            .collect();
        Self {
            storage,
            nbits: self.nbits,
        }
        }
    };
}

// convert total bit to length
// input: Number of bits
// output:
//
// 1. the number of Vector used
// 2. after filling 1, the remaining bytes should be filled
// 3. after filling 2, the remaining bits should be filled
//
// notice that this result represents the length of vector
// so if 3. is 0, it means no extra bits after filling bytes
// return (length of storage, u64 of last container, bit of last elem)
// any bits > length of last elem should be set to 0
#[inline]
fn bit_to_len(nbits: usize) -> (usize, usize, usize) {
    (nbits / (BIT_WIDTH as usize), (nbits % (BIT_WIDTH as usize)) / 64, nbits % 64)
}

#[inline]
fn set_bit(flag: bool, bytes: u64, offset: u32) -> u64 {
    match flag {
        true => bytes | (1u64 << offset),
        false => bytes & !(1u64 << offset),
    }
}

impl BitVec {
    /// Create an empty bitvec with `nbits` initial elements.
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::zeros(10);
    /// assert_eq!(bitvec.len(), 10);
    /// ```
    pub fn zeros(nbits: usize) -> Self {
        let len = (nbits + BIT_WIDTH - 1) / BIT_WIDTH;
        let storage = (0..len).map(|_| BitContainer::splat(0)).collect();
        Self { storage, nbits }
    }

    /// Create a bitvec containing all 0 .. nbits elements.
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::ones(10);
    /// assert_eq!(bitvec.len(), 10);
    /// ```
    pub fn ones(nbits: usize) -> Self {
        let (len, bytes, bits) = bit_to_len(nbits);
        let mut storage = (0..len)
            .map(|_| BitContainer::splat(u64::MAX))
            .collect::<Vec<_>>();
        if bytes > 0 || bits > 0 {
            let slice = (0..bytes as u64)
                .map(|_| u64::MAX)
                .chain([(u64::MAX << (u64::BITS - bits as u32) >> (u64::BITS - bits as u32))])
                .chain((0..(BIT_WIDTH as u32 / u64::BITS) - bytes as u32 - 1).map(|_| 0))
                .collect::<Vec<_>>();
            assert_eq!(slice.len(), 4);
            storage.push(BitContainer::from(&slice[0..4]));
        }
        Self { storage, nbits }
    }

    /// Create a bitvec from an Iterator of bool.
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::from_bool_iterator((0..10).map(|x| x % 2 == 0));
    /// assert_eq!(bitvec.len(), 10);
    /// assert_eq!(<BitVec as Into<Vec<bool>>>::into(bitvec), vec![true, false, true, false, true, false, true, false, true, false]);
    ///
    /// let bitvec = BitVec::from_bool_iterator((0..1000).map(|x| x < 50));
    /// assert_eq!(bitvec.len(), 1000);
    /// assert_eq!(bitvec.get(49), Some(true));
    /// assert_eq!(bitvec.get(50), Some(false));
    /// assert_eq!(bitvec.get(999), Some(false));
    /// assert_eq!(<BitVec as Into<Vec<bool>>>::into(bitvec), (0..1000).map(|x| x<50).collect::<Vec<bool>>());
    /// ```
    pub fn from_bool_iterator<I: Iterator<Item = bool>>(i: I) -> Self {
        // FIXME: any better implementation?
        let mut storage = Vec::new();
        let mut current_slice = [0u64; 4];
        let mut nbits = 0;
        for b in i {
            if b {
                current_slice[nbits % BIT_WIDTH / 64] |= 1 << (nbits % 64);
            }
            nbits += 1;
            if nbits % BIT_WIDTH == 0 {
                storage.push(BitContainer::from(current_slice));
                current_slice = [0u64; 4];
            }
        }
        if nbits % BIT_WIDTH > 0 {
            storage.push(BitContainer::from(current_slice));
        }
        Self { storage, nbits }
    }

    /// Initialize from a set of integers.
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::from_slice(&[0,5,9]);
    /// assert_eq!(<BitVec as Into<Vec<bool>>>::into(bitvec), vec![true, false, false, false, false, true, false, false, false, true]);
    /// ```
    pub fn from_slice(slice: &[usize]) -> Self {
        let mut bv = BitVec::zeros(slice.len());
        for i in slice {
            bv.set(*i, true);
        }
        bv
    }

    /// Initialize from a raw u64 slice.
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::from_slice_raw(&[3]);
    /// assert_eq!(bitvec.get(0), Some(true));
    /// assert_eq!(bitvec.get(1), Some(true));
    /// ```
    pub fn from_slice_raw(slice: &[u64]) -> Self {
        let iter = &mut slice.iter();
        let mut storage = Vec::with_capacity((slice.len() + 3) / 4);

        while let Some(a0) = iter.next() {
            let a1 = match iter.next() {
                Some(value) => value,
                _ => &0u64,
            };
            let a2 = match iter.next() {
                Some(value) => value,
                _ => &0u64,
            };
            let a3 = match iter.next() {
                Some(value) => value,
                _ => &0u64,
            };
            storage.push(BitContainer::from([*a0, *a1, *a2, *a3]));
        }

        Self { storage, nbits: slice.len() * 64 }
    }

    /// Length of this bitvec.
    ///
    /// To get the number of elements, use `count_ones`
    pub fn len(&self) -> usize {
        self.nbits
    }

    /// Remove or add `index` to the set.
    /// If index > self.len, the bitvec will be expanded to `index`.
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let mut bitvec = BitVec::zeros(10);
    /// assert_eq!(bitvec.len(), 10);
    /// bitvec.set(15, true);  
    /// // now 15 has been added to the set, its total len is 16.
    /// assert_eq!(bitvec.len(), 16);
    /// assert_eq!(bitvec.get(15), Some(true));
    /// assert_eq!(bitvec.get(14), Some(false));
    /// ```
    pub fn set(&mut self, index: usize, flag: bool) {
        let (i, bytes, bits) = bit_to_len(index);
        if self.nbits <= index {
            let i = if bytes > 0 || bits > 0 { i + 1 } else { i };
            self.storage
                .extend((0..i - self.storage.len()).map(move |_| BitContainer::ZERO));
            self.nbits = index + 1;
        }
        let mut arr = self.storage[i].to_array();
        arr[bytes] = set_bit(flag, arr[bytes], bits as u32);
        self.storage[i] = BitContainer::from(arr);
    }

    /// Set all items in bitvec to false
    pub fn set_all_false(&mut self) {
        self.storage.iter_mut().for_each(move |x| *x = BitContainer::ZERO);
    }

    /// Set all items in bitvec to true
    pub fn set_all_true(&mut self) {
        let (_, bytes, bits) = bit_to_len(self.nbits);
        self.storage.iter_mut().for_each(move |x| *x = BitContainer::splat(u64::MAX));
        if bytes > 0 || bits > 0 {
            let slice = (0..bytes as u64)
                .map(|_| u64::MAX)
                .chain([(u64::MAX << (u64::BITS - bits as u32) >> (u64::BITS - bits as u32))])
                .chain((0..(BIT_WIDTH as u32 / u64::BITS) - bytes as u32 - 1).map(|_| 0))
                .collect::<Vec<_>>();
            assert_eq!(slice.len(), 4);

            // unwrap here is safe since bytes > 0 || bits > 0 => self.nbits > 0
            *(self.storage.last_mut().unwrap()) = BitContainer::from(&slice[0..4]);
        }
    }

    /// Set all items in bitvec to flag
    pub fn set_all(&mut self, flag: bool) {
        match flag {
            true => self.set_all_true(),
            false => self.set_all_false(),
        }
    }

    /// Check if `index` exists in current set.
    ///
    /// * If exists, return `Some(true)`
    /// * If index < current.len and element doesn't exist, return `Some(false)`.
    /// * If index >= current.len, return `None`.
    ///
    /// Examlpe:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec : BitVec = (0 .. 15).map(|x| x%3 == 0).into();
    /// assert_eq!(bitvec.get(3), Some(true));
    /// assert_eq!(bitvec.get(5), Some(false));
    /// assert_eq!(bitvec.get(14), Some(false));
    /// assert_eq!(bitvec.get(15), None);
    /// ```
    pub fn get(&self, index: usize) -> Option<bool> {
        if self.nbits <= index {
            None
        } else {
            let (index, bytes, bits) = bit_to_len(index);
            Some((self.storage[index].to_array()[bytes] & (1u64 << bits)) > 0)
        }
    }

    /// Directly return a `bool` instead of an `Option`
    ///
    /// * If exists, return `true`.
    /// * If doesn't exist, return false.
    /// * If index >= current.len, panic.
    ///
    ///
    /// Examlpe:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec : BitVec = (0 .. 15).map(|x| x%3 == 0).into();
    /// assert_eq!(bitvec.get_unchecked(3), true);
    /// assert_eq!(bitvec.get_unchecked(5), false);
    /// assert_eq!(bitvec.get_unchecked(14), false);
    /// ```
    pub fn get_unchecked(&self, index: usize) -> bool {
        if self.nbits <= index {
            panic!("index out of bounds {} > {}", index, self.nbits);
        } else {
            let (index, bytes, bits) = bit_to_len(index);
            (self.storage[index].to_array()[bytes] & (1u64 << bits)) > 0
        }
    }

    impl_operation!(and, and_cloned, &);
    impl_operation!(or, or_cloned, |);
    impl_operation!(xor, xor_cloned, ^);

    /// difference operation
    ///
    /// `A.difference(B)` calculates `A\B`, e.g.
    ///
    /// ```text
    /// A = [1,2,3], B = [2,4,5]
    /// A\B = [1,3]
    /// ```
    ///
    /// also notice that
    ///
    /// ```text
    /// A.difference(B) | B.difference(A) == A ^ B
    /// ```
    /// 
    /// Example:
    /// 
    /// ```rust
    /// use bitvec_simd::BitVec;
    /// 
    /// let bitvec: BitVec = (0 .. 5_000).map(|x| x % 2 == 0).into();
    /// let bitvec2 : BitVec = (0 .. 5_000).map(|x| x % 3 == 0).into();
    /// assert_eq!(bitvec.difference_cloned(&bitvec2) | bitvec2.difference_cloned(&bitvec), bitvec.xor_cloned(&bitvec2));
    /// let bitvec3 : BitVec = (0 .. 5_000).map(|x| x % 2 == 0 && x % 3 != 0).into();
    /// assert_eq!(bitvec.difference(bitvec2), bitvec3);
    /// ```
    pub fn difference(self, other: Self) -> Self {
        self.and(other.not())
    }

    pub fn difference_cloned(&self, other: &Self) -> Self {
        // FIXME: This implementation has one extra clone
        self.and_cloned(&other.clone().not())
    }

    // not should make sure bits > nbits is 0
    /// inverse every bits in the vector.
    ///
    /// If your bitvec have len `1_000` and contains `[1,5]`,
    /// after inverse it will contains `0, 2..=4, 6..=999`
    pub fn inverse(self) -> Self {
        let (i, bytes, bits) = bit_to_len(self.nbits);
        let mut storage = self.storage.into_iter().map(|x| !x).collect::<Vec<_>>();
        if bytes > 0 || bits > 0 {
            assert_eq!(storage.len(), i + 1);
            if let Some(s) = storage.get_mut(i) {
                let mut arr = s.to_array();
                arr[bytes] = arr[bytes] << (u64::BITS - bits as u32) >> (u64::BITS - bits as u32);
                for index in (bytes + 1)..4 {
                    arr[index] = 0;
                }
                *s = arr.into();
            } else {
                panic!("incorrect internal representation of self")
            }
        }

        Self {
            storage,
            nbits: self.nbits,
        }
    }


    /// Count the number of elements existing in this bitvec.
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    /// 
    /// let bitvec: BitVec = (0..10_000).map(|x| x%2==0).into();
    /// assert_eq!(bitvec.count_ones(), 5000);
    ///
    /// let bitvec: BitVec = (0..30_000).map(|x| x%3==0).into();
    /// assert_eq!(bitvec.count_ones(), 10_000);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.storage
            .iter()
            .map(|x| {
                x.to_array().iter().map(|a| a.count_ones()).sum::<u32>()
            }
            )
            .sum::<u32>() as usize
    }

    /// Count the number of elements existing in this bitvec, before the specified index.
    /// Panics if index is invalid.
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    /// 
    /// let bitvec: BitVec = (0..10_000).map(|x| x%2==0).into();
    /// assert_eq!(bitvec.count_ones_before(5000), 2500);
    ///
    /// let bitvec: BitVec = (0..30_000).map(|x| x%3==0).into();
    /// assert_eq!(bitvec.count_ones_before(10000), 3334);
    /// ```
    pub fn count_ones_before(&self, index: usize) -> usize {
        assert!(index <= self.nbits);
        let (i, bytes, bits) = bit_to_len(index - 1);
        let mut ones = self.storage.iter().take(i).map(|x| {
            x.to_array().iter().map(|a| a.count_ones()).sum::<u32>()
        }).sum::<u32>();
        if bytes > 0 || bits > 0 {
            // Safe unwrap here
            let arr = self.storage.iter().skip(i).next().unwrap().to_array();
            ones += arr.iter().take(bytes).map(|x| x.count_ones()).sum::<u32>();
            if bits > 0 {
                let x = arr.iter().skip(bytes).next().unwrap();
                ones += (x & ((1u64 << (bits + 1)) - 1)).count_ones();
            }
        }
        ones as usize
    }

    /// return true if contains at least 1 element
    pub fn any(&self) -> bool {
        self.storage
            .iter()
            .any(|x| x.to_array().iter().map(|a| a.count_ones()).sum::<u32>() > 0)
    }

    /// return true if contains self.len elements
    pub fn all(&self) -> bool {
        self.count_ones() == self.nbits
    }

    /// return true if set is empty
    pub fn none(&self) -> bool {
        !self.any()
    }

    /// Return true if set is empty.
    /// Totally the same with `self.none()`
    pub fn is_empty(&self) -> bool {
        !self.any()
    }

    /// Consume self and generate a `Vec<bool>` with length == self.len().
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::from_bool_iterator((0..10).map(|i| i % 3 == 0));
    /// let bool_vec = bitvec.into_bools();
    /// assert_eq!(bool_vec, vec![true, false, false, true, false, false, true, false, false, true])
    /// ```
    pub fn into_bools(self) -> Vec<bool> {
        self.into()
    }

    /// Consume self and geterate a `Vec<usize>` which only contains the number exists in this set.
    ///
    /// Example:
    ///
    /// ```rust
    /// use bitvec_simd::BitVec;
    ///
    /// let bitvec = BitVec::from_bool_iterator((0..10).map(|i| i%3 == 0));
    /// let usize_vec = bitvec.into_usizes();
    /// assert_eq!(usize_vec, vec![0,3,6,9]);
    /// ```
    pub fn into_usizes(self) -> Vec<usize> {
        self.into()
    }
}

impl<I: Iterator<Item = bool>> From<I> for BitVec {
    fn from(i: I) -> Self {
        Self::from_bool_iterator(i)
    }
}

impl From<BitVec> for Vec<bool> {
    fn from(v: BitVec) -> Self {
        v.storage
            .into_iter()
            .flat_map(|x| Into::<[u64; 4]>::into(x))
            .flat_map(|x| (0..u64::BITS).map(move |i| (x >> i) & 1 > 0))
            .take(v.nbits)
            .collect()
    }
}

impl From<BitVec> for Vec<usize> {
    fn from(v: BitVec) -> Self {
        v.storage
            .into_iter()
            .flat_map(|x| Into::<[u64; 4]>::into(x))
            .flat_map(|x| (0..u64::BITS).map(move |i| (x >> i) & 1 > 0))
            .take(v.nbits)
            .enumerate()
            .filter(|(_, b)| *b)
            .map(|(i, _)| i)
            .collect()
    }
}

impl Index<usize> for BitVec {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        if self.get_unchecked(index) {
            &true
        } else {
            &false
        }
    }
}

impl Display for BitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (i, bytes, bits) = bit_to_len(self.nbits);
        for index in 0..i {
            let s = self.storage[index];
            for u in 0..8 {
                write!(f, "{:064b} ", s.to_array()[u])?;
            }
        }
        if bytes > 0 || bits > 0 {
            let s = self.storage[i];
            for u in 0..bytes {
                write!(f, "{:064b} ", s.to_array()[u])?;
            }
            write!(f, "{:064b}", s.to_array()[bytes])
        } else {
            Ok(())
        }
    }
}

impl PartialEq for BitVec {
    // eq should always ignore the bits > nbits
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.nbits, other.nbits);
        self.storage
            .iter()
            .zip(other.storage.iter())
            .all(|(a, b)| a == b)
    }
}

impl BitAnd for BitVec {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}

impl BitOr for BitVec {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl BitXor for BitVec {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

impl Not for BitVec {
    type Output = Self;
    fn not(self) -> Self::Output {
        self.inverse()
    }
}

#[test]
fn test_bit_to_len() {
    // contain nothing
    assert_eq!(bit_to_len(0), (0, 0, 0));
    assert_eq!(bit_to_len(1), (0, 0, 1));
    // 64bit only stores in a u64
    assert_eq!(bit_to_len(64), (0, 1, 0));
    // extra bit stores in extra u64
    assert_eq!(bit_to_len(65), (0, 1, 1));
    // BIT_WIDTH bit only stores in a vector
    assert_eq!(bit_to_len(BIT_WIDTH), (1, 0, 0));
    assert_eq!(bit_to_len(BIT_WIDTH + 1), (1, 0, 1));
    assert_eq!(bit_to_len(BIT_WIDTH + 65), (1, 1, 1));
}

#[test]
fn test_bit_vec_count_ones() {
    let mut bitvec = BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
    assert_eq!(bitvec.count_ones_before(500), 500);
    bitvec.set(1500, true);
    assert_eq!(bitvec.count_ones(), 1001);
    assert_eq!(bitvec.count_ones_before(500), 500);
}

#[test]
fn test_bit_vec_all_any() {
    let mut bitvec = BitVec::ones(1000);
    assert!(bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    bitvec.set(10, false);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    bitvec.set(1500, true);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    let mut bitvec = BitVec::zeros(1000);
    assert!(!bitvec.all());
    assert!(!bitvec.any());
    assert!(bitvec.none());
    bitvec.set(1500, true);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
}

#[test]
fn test_bitvec_and_xor() {
    let bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    let bitvec3 = BitVec::zeros(1000);
    assert_eq!(bitvec.xor_cloned(&bitvec2), BitVec::zeros(1000));
    assert_eq!(bitvec.xor_cloned(&bitvec3), BitVec::ones(1000));
    assert_eq!(bitvec ^ bitvec2, BitVec::zeros(1000));
    let mut bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec ^ bitvec2;
    assert!(bitvec3[400]);
    assert_eq!(bitvec3.count_ones(), 1);
}

#[test]
fn test_bitvec_and_or() {
    let bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    let bitvec3 = BitVec::zeros(1000);
    assert_eq!(bitvec.or_cloned(&bitvec2), BitVec::ones(1000));
    assert_eq!(bitvec.or_cloned(&bitvec3), BitVec::ones(1000));
    assert_eq!(bitvec | bitvec2, BitVec::ones(1000));
    let mut bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec | bitvec2;
    assert!(bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count_ones(), 1000);
}

#[test]
fn test_bitvec_and_and() {
    let bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    let bitvec3 = BitVec::zeros(1000);
    assert_eq!(bitvec.and_cloned(&bitvec2), BitVec::ones(1000));
    assert_eq!(bitvec.and_cloned(&bitvec3), BitVec::zeros(1000));
    assert_eq!(bitvec & bitvec2, BitVec::ones(1000));
    let mut bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec & bitvec2;
    assert!(!bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count_ones(), 1000 - 1);
}

#[test]
fn test_bitvec_not() {
    let bitvec = BitVec::ones(1000);
    assert_eq!(bitvec, BitVec::ones(1000));
    assert_eq!(bitvec.not(), BitVec::zeros(1000));
}

#[test]
fn test_bitvec_eq() {
    let mut bitvec = BitVec::ones(1000);
    assert_eq!(bitvec, BitVec::ones(1000));
    assert_ne!(bitvec, BitVec::zeros(1000));
    bitvec.set(50, false);
    assert_ne!(bitvec, BitVec::ones(1000));
    bitvec.set(50, true);
    assert_eq!(bitvec, BitVec::ones(1000));
}

#[test]
fn test_bitvec_creation() {
    let mut bitvec = BitVec::zeros(1000);
    for i in 0..1500 {
        if i < 1000 {
            assert_eq!(bitvec.get(i), Some(false));
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
    bitvec.set(900, true);
    for i in 0..1500 {
        if i < 1000 {
            if i == 900 {
                assert_eq!(bitvec.get(i), Some(true));
            } else {
                assert_eq!(bitvec.get(i), Some(false));
            }
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
    bitvec.set(1300, true);
    for i in 0..1500 {
        if i <= 1300 {
            if i == 900 || i == 1300 {
                assert_eq!(bitvec.get(i), Some(true));
            } else {
                assert_eq!(bitvec.get(i), Some(false));
            }
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }

    let bitvec = BitVec::from_slice_raw(&[3]);
    assert_eq!(bitvec.get(0), Some(true));
    assert_eq!(bitvec.get(1), Some(true));
    assert_eq!(bitvec.get(2), Some(false));
    assert_eq!(bitvec.get(63), Some(false));
    assert_eq!(bitvec.get(64), None);
}


#[test]
fn test_bitvec_set_all() {
    let mut bitvec = BitVec::zeros(1000);
    bitvec.set_all(true);
    for i in 0..1500 {
        if i < 1000 {
            assert_eq!(bitvec.get(i), Some(true));
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
    bitvec.set_all(false);
    for i in 0..1500 {
        if i < 1000 {
            assert_eq!(bitvec.get(i), Some(false));
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
}
