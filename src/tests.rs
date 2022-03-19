use crate::*;

#[test]
fn test_bit_to_len() {
    type T = BitVecSimd<[u64x4; 4], 4>;
    // contain nothing
    assert_eq!(T::bit_to_len(0), (0, 0, 0));
    assert_eq!(T::bit_to_len(1), (0, 0, 1));
    // 64bit only stores in a u64
    assert_eq!(T::bit_to_len(64), (0, 1, 0));
    // extra bit stores in extra u64
    assert_eq!(T::bit_to_len(65), (0, 1, 1));
    // BIT_WIDTH bit only stores in a vector
    assert_eq!(T::bit_to_len(u64x4::BIT_WIDTH), (1, 0, 0));
    assert_eq!(T::bit_to_len(u64x4::BIT_WIDTH + 1), (1, 0, 1));
    assert_eq!(T::bit_to_len(u64x4::BIT_WIDTH + 65), (1, 1, 1));
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
fn test_bit_vec_leading_zeros() {
    let mut bitvec = BitVec::zeros(1000);
    assert_eq!(bitvec.leading_zeros(), 1000);
    bitvec.set(499, true);
    assert_eq!(bitvec.leading_zeros(), 500);
    bitvec.set(599, true);
    assert_eq!(bitvec.leading_zeros(), 400);
    bitvec.set(699, true);
    assert_eq!(bitvec.leading_zeros(), 300);
    bitvec.set(799, true);
    assert_eq!(bitvec.leading_zeros(), 200);
    bitvec.set(899, true);
    assert_eq!(bitvec.leading_zeros(), 100);
    bitvec.set(999, true);
    assert_eq!(bitvec.leading_zeros(), 0);

    bitvec = BitVecSimd::zeros(10);
    bitvec.set(3, true);
    assert_eq!(bitvec.leading_zeros(), 6);
}

#[test]
fn test_bit_vec_resize() {
    for i in (0..3333).filter(|x| x % 13 == 0) {
        for j in (0..6666).filter(|x| x % 37 == 0) {
            let mut b = BitVec::ones(i);
            b.resize(j, true);
            assert_eq!(b.len(), j);
            assert_eq!(b.count_ones(), j);
        }
    }

    let mut bitvec = BitVec::ones(3333);
    for i in 3333..6666 {
        bitvec.resize(i, false);
        assert_eq!(bitvec.len(), i);
    }
    for i in 3333..0 {
        bitvec.resize(i, false);
        assert_eq!(bitvec.len(), i);
        assert_eq!(bitvec.count_ones(), i);
    }
}

#[test]
fn test_bit_vec_shrink_to() {
    let mut bitvec = BitVec::ones(3333);
    bitvec.shrink_to(2222);
    assert_eq!(bitvec.len(), 2222);
}

#[test]
#[should_panic]
fn test_bit_vec_shrink_to_painc() {
    let mut bitvec = BitVec::ones(3333);
    bitvec.shrink_to(4444);
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
fn test_bitvec_xor() {
    let mut bitvec = BitVec::ones(1000);
    let mut bitvec2 = BitVec::ones(1000);
    let bitvec3 = BitVec::zeros(1000);
    assert_eq!(bitvec.xor_cloned(&bitvec2), BitVec::zeros(1000));
    assert_eq!(bitvec.xor_cloned(&bitvec3), BitVec::ones(1000));
    assert_eq!(&bitvec ^ &bitvec2, BitVec::zeros(1000));
    assert_eq!((&mut bitvec) ^ &bitvec2, BitVec::zeros(1000));
    assert_eq!(&bitvec ^ (&mut bitvec2), BitVec::zeros(1000));
    assert_eq!((&mut bitvec) ^ (&mut bitvec2), BitVec::zeros(1000));
    assert_eq!(&bitvec ^ bitvec2.clone(), BitVec::zeros(1000));
    assert_eq!((&mut bitvec) ^ bitvec2.clone(), BitVec::zeros(1000));
    assert_eq!(bitvec.clone() ^ &bitvec2, BitVec::zeros(1000));
    assert_eq!(bitvec.clone() ^ (&mut bitvec2), BitVec::zeros(1000));
    assert_eq!(bitvec ^ bitvec2, BitVec::zeros(1000));

    let mut bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec ^ bitvec2;
    assert!(bitvec3[400]);
    assert_eq!(bitvec3.count_ones(), 1);

    let mut bitvec = BitVec::ones(1000);
    bitvec ^= BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 0);
    bitvec ^= &BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
    bitvec ^= &mut BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 0);
}

#[test]
fn test_bitvec_or() {
    let mut bitvec = BitVec::ones(1000);
    let mut bitvec2 = BitVec::ones(1000);
    let bitvec3 = BitVec::zeros(1000);
    assert_eq!(bitvec.or_cloned(&bitvec2), BitVec::ones(1000));
    assert_eq!(bitvec.or_cloned(&bitvec3), BitVec::ones(1000));
    assert_eq!(&bitvec | &bitvec2, BitVec::ones(1000));
    assert_eq!((&mut bitvec) | &bitvec2, BitVec::ones(1000));
    assert_eq!(&bitvec | (&mut bitvec2), BitVec::ones(1000));
    assert_eq!((&mut bitvec) | (&mut bitvec2), BitVec::ones(1000));
    assert_eq!(&bitvec | bitvec2.clone(), BitVec::ones(1000));
    assert_eq!((&mut bitvec) | bitvec2.clone(), BitVec::ones(1000));
    assert_eq!(bitvec.clone() | &bitvec2, BitVec::ones(1000));
    assert_eq!(bitvec.clone() | (&mut bitvec2), BitVec::ones(1000));
    assert_eq!(bitvec | bitvec2, BitVec::ones(1000));
    let mut bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec | bitvec2;
    assert!(bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count_ones(), 1000);

    let mut bitvec = BitVec::ones(1000);
    bitvec |= BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
    bitvec |= &BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
    bitvec |= &mut BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
}

#[test]
fn test_bitvec_and() {
    let mut bitvec = BitVec::ones(1000);
    let mut bitvec2 = BitVec::ones(1000);
    let bitvec3 = BitVec::zeros(1000);
    assert_eq!(bitvec.and_cloned(&bitvec2), BitVec::ones(1000));
    assert_eq!(bitvec.and_cloned(&bitvec3), BitVec::zeros(1000));
    assert_eq!(&bitvec & &bitvec2, BitVec::ones(1000));
    assert_eq!((&mut bitvec) & &bitvec2, BitVec::ones(1000));
    assert_eq!(&bitvec & (&mut bitvec2), BitVec::ones(1000));
    assert_eq!((&mut bitvec) & (&mut bitvec2), BitVec::ones(1000));
    assert_eq!(&bitvec & bitvec2.clone(), BitVec::ones(1000));
    assert_eq!((&mut bitvec) & bitvec2.clone(), BitVec::ones(1000));
    assert_eq!(bitvec.clone() & &bitvec2, BitVec::ones(1000));
    assert_eq!(bitvec.clone() & (&mut bitvec2), BitVec::ones(1000));
    assert_eq!(bitvec & bitvec2, BitVec::ones(1000));
    let mut bitvec = BitVec::ones(1000);
    let bitvec2 = BitVec::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec & bitvec2;
    assert!(!bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count_ones(), 1000 - 1);

    let mut bitvec = BitVec::ones(1000);
    bitvec &= BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
    bitvec &= &BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
    bitvec &= &mut BitVec::ones(1000);
    assert_eq!(bitvec.count_ones(), 1000);
}

#[test]
fn test_bitvec_not() {
    let mut bitvec = BitVec::ones(1000);
    assert_eq!(bitvec, BitVec::ones(1000));
    assert_eq!((&bitvec).not(), BitVec::zeros(1000));
    assert_eq!((&mut bitvec).not(), BitVec::zeros(1000));
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

    assert!(&bitvec == BitVec::ones(1000));
    assert!(&bitvec == &BitVec::ones(1000));
    assert!(&bitvec == (&mut BitVec::ones(1000)));
    assert!((&mut bitvec) == BitVec::ones(1000));
    assert!((&mut bitvec) == &BitVec::ones(1000));
    assert!((&mut bitvec) == (&mut BitVec::ones(1000)));
    assert!(bitvec == BitVec::ones(1000));
    assert!(bitvec == &BitVec::ones(1000));
    assert!(bitvec == (&mut BitVec::ones(1000)));
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

    let bitvec = BitVec::from_slice_copy(&[7], 2);
    assert_eq!(bitvec.get(0), Some(true));
    assert_eq!(bitvec.get(1), Some(true));
    assert_eq!(bitvec.get(2), None);

    let bitvec = BitVec::from_slice_copy(&[7], 64);
    assert_eq!(bitvec.get(0), Some(true));
    assert_eq!(bitvec.get(1), Some(true));
    assert_eq!(bitvec.get(2), Some(true));
    assert_eq!(bitvec.get(3), Some(false));
    assert_eq!(bitvec.get(63), Some(false));
    assert_eq!(bitvec.get(64), None);

    let v = vec![7];
    let buf = v.as_ptr();
    let bitvec = unsafe { BitVec::from_raw_copy(buf, 1, 2) };
    assert_eq!(v.len(), 1); // ensure v lives long enough
    assert_eq!(bitvec.get(0), Some(true));
    assert_eq!(bitvec.get(1), Some(true));
    assert_eq!(bitvec.get(2), None);

    let v = vec![7];
    let buf = v.as_ptr();
    let bitvec = unsafe { BitVec::from_raw_copy(buf, 1, 64) };
    assert_eq!(v.len(), 1); // ensure v lives long enough
    assert_eq!(bitvec.get(0), Some(true));
    assert_eq!(bitvec.get(1), Some(true));
    assert_eq!(bitvec.get(2), Some(true));
    assert_eq!(bitvec.get(3), Some(false));
    assert_eq!(bitvec.get(63), Some(false));
    assert_eq!(bitvec.get(64), None);
}

#[test]
fn test_bitvec_set_raw_copy() {
    let v = vec![7];
    let buf = v.as_ptr();
    let mut bitvec = unsafe { BitVec::from_raw_copy(buf, 1, 64) };
    let ptr = bitvec.storage.as_mut_ptr();
    let buffer_len = bitvec.storage.len();
    let mut bitvec2 = BitVec::zeros(1);
    unsafe {
        bitvec2.set_raw_copy(ptr, buffer_len, bitvec.nbits);
    }
    assert_eq!(v.len(), 1); // ensure v lives long enough
    assert_eq!(bitvec2.get(0), Some(true));
    assert_eq!(bitvec2.get(1), Some(true));
    assert_eq!(bitvec2.get(2), Some(true));
    assert_eq!(bitvec2.get(3), Some(false));
    assert_eq!(bitvec2.get(63), Some(false));
    assert_eq!(bitvec2.get(64), None);
}

#[test]
fn test_bitvec_set_raw() {
    let mut bitvec = BitVec::ones(1025);
    let ptr = bitvec.storage.as_mut_ptr();
    let buffer_len = bitvec.storage.len();
    let capacity = bitvec.storage.capacity();
    let nbits = bitvec.nbits;
    let spilled = bitvec.storage.spilled();
    let mut bitvec2 = BitVec::zeros(1);
    unsafe {
        std::mem::forget(bitvec);
        assert!(spilled);
        bitvec2.set_raw(ptr, buffer_len, capacity, nbits);
        assert_eq!(bitvec2.get(0), Some(true));
        assert_eq!(bitvec2.get(1024), Some(true));
        assert_eq!(bitvec2.get(1025), None);
    }
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

#[test]
fn test_bitvec_display() {
    let bitvec = BitVec::ones(5);
    assert_eq!(format!("{}", bitvec), "11111");
}

#[cfg(feature = "use_serde")]
#[test]
fn test_ser_de() {
    let bitvec = BitVec::ones(5);
    serde_test::assert_tokens(
        &bitvec,
        &[
            serde_test::Token::Struct {
                name: "BitVecSimd",
                len: 2,
            },
            serde_test::Token::Str("storage"),
            serde_test::Token::Seq { len: Some(1) },
            serde_test::Token::U64(31),
            serde_test::Token::SeqEnd,
            serde_test::Token::Str("nbits"),
            serde_test::Token::U64(5),
            serde_test::Token::StructEnd,
        ],
    );
}
