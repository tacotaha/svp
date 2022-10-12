# SVP

Lattice sieving over the integers with arbitrary precision.

Includes the sampling algorithm described by [ \[GPV08\]](https://eprint.iacr.org/2007/432)
and the Gauss Sieve described in [ \[MV10\]](https://eccc.weizmann.ac.il//report/2009/065/).

# Example

```rust
use svp::*;

// LLL/BKZ reduced basis
let mut b = vec![
    nvec![
        Integer::new(),
        Integer::new(),
        Integer::new(),
        Integer::from(-1),
        Integer::from(-1)
    ],
    nvec![
        Integer::from(-1),
        Integer::from(-2),
        Integer::from(-2),
        Integer::from(-1),
        Integer::from(1)
    ],
    nvec![
        Integer::from(3),
        Integer::new(),
        Integer::from(-1),
        Integer::from(-1),
        Integer::from(1)
    ],
    nvec![
        Integer::from(-1),
        Integer::from(3),
        Integer::from(-2),
        Integer::from(2),
        Integer::from(-2)
    ],
    nvec![
        Integer::from(2),
        Integer::from(-3),
        Integer::from(1),
        Integer::from(2),
        Integer::from(-2)
    ],
];

// Compute squared norms
for i in 0..b.len() {
    b[i].norm = Some(&b[i] * &b[i]);
}

// Init lattice
let l = Lattice { basis: b.clone() };

// Rejection sampling parameter
let t = Float::with_val(53, b.len()).ln();

// Init Gasuss Sieve
let mut gs = gsieve![l, t];

// Short vectors sorted in ascending order
let short_vecs = gs.sieve();
```
