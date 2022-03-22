# bitvec\_simd

[![GHA Status]][GitHub Actions] [![Latest Version]][crates.io] [![Documentation]][docs.rs] ![License] [![Minimum Supported Rust Version]][Rust 1.56]

A rust library for bit vector, featuring:
- SIMD accelerated via [wide](https://crates.io/crates/wide).
- Serialize and deserialize via [serde](https://crates.io/crates/serde).
- Stack allocation when bit count is small via [smallvec](https://crates.io/crates/smallvec).
- `#![no_std]` support.
- *Many* performance optimisations.

## Usage

Add `bitvec_simd` to `Cargo.toml`:

```toml
[dependencies]
bitvec_simd = "0.20"
```

If you want [serde](https://crates.io/crates/serde) support, include the feature like this:

```toml
[dependencies]
bitvec_simd = { version = "0.20", features = ["serde"] }
```

If you want to use bitvec_simd in a program that has `#![no_std]`, just drop default features:

```toml
[dependencies]
bitvec_simd = { version = "0.20", default-features = false }
```

### Example

```rust

let mut bitvec = BitVec::ones(1000); // create a bitvec contains 0 ..= 999
bitvec.set(900, false); // delete 900 from bitvec
bitvec.set(1200, true); // add 1200 to bitvec (and expand bitvec to length 1201)
let bitvec2 = BitVec::ones(1000);

let new_bitvec = bitvec.and_cloned(&bitvec2); // and operation, without consume
let new_bitvec2 = bitvec & bitvec2; // and operation, consume both bitvec

// Operation Supported:
// and, or, xor, not, eq, eq_left

assert_eq!(new_bitvec, new_bitvec2);
```

## Performance 

Compared on AMD Ryzen 7 3700X, aginst:

* [bit\_vec 0.6.3](https://crates.io/crates/bit-vec)
* [bitvec 1.0.0](https://crates.io/crates/bitvec)

```
$ cargo bench       

bitvec_simd(this crate) time:   [465.32 ns 468.04 ns 471.35 ns]
Found 7 outliers among 100 measurements (7.00%)
  2 (2.00%) high mild
  5 (5.00%) high severe

bit-vec 0.6             time:   [1.9115 us 1.9129 us 1.9147 us]
                        change: [+0.0742% +0.2796% +0.5452%] (p = 0.01 < 0.05)
                        Change within noise threshold.
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) high mild
  3 (3.00%) high severe

bitvec 1.0              time:   [598.79 us 599.30 us 599.99 us]
                        change: [-0.1615% +0.1525% +0.3552%] (p = 0.30 > 0.05)
                        No change in performance detected.
Found 14 outliers among 100 measurements (14.00%)
  1 (1.00%) high mild
  13 (13.00%) high severe

bitvec_simd(this crate) with creation
                        time:   [1.6427 us 1.6455 us 1.6491 us]
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high severe

bit-vec 0.6 with creation
                        time:   [2.4678 us 2.4696 us 2.4716 us]
                        change: [+1.1134% +1.3933% +1.6146%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 4 outliers among 100 measurements (4.00%)
  3 (3.00%) high mild
  1 (1.00%) high severe

bitvec 1.0 with creation
                        time:   [220.12 us 220.81 us 221.71 us]
                        change: [-0.7438% -0.2211% +0.1436%] (p = 0.41 > 0.05)
                        No change in performance detected.
Found 9 outliers among 100 measurements (9.00%)
  5 (5.00%) high mild
  4 (4.00%) high severe
```

##

## Reference

Some code of this crate is from (https://github.com/horasal/bitvector_simd)

## License

This project is licensed under [MIT License](https://opensource.org/licenses/MIT)
  ([LICENSE-MIT](https://github.com/GCCFeli/bitvec_simd/blob/master/LICENSE)).

[GHA Status]: https://github.com/GCCFeli/bitvec_simd/actions/workflows/rust.yml/badge.svg?event=push
[GitHub Actions]: https://github.com/GCCFeli/bitvec_simd/actions
[crates.io]: https://crates.io/crates/bitvec_simd
[Latest Version]: https://img.shields.io/crates/v/bitvec_simd.svg
[Documentation]: https://docs.rs/bitvec_simd/badge.svg
[docs.rs]: https://docs.rs/bitvec_simd
[License]: https://img.shields.io/crates/l/bitvec_simd.svg
[Minimum Supported Rust Version]: https://img.shields.io/badge/Rust-1.56+-blue?color=fc8d62&logo=rust
[Rust 1.56]: https://github.com/rust-lang/rust/blob/master/RELEASES.md#version-1561-2021-11-01

