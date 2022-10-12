use crate::{GaussReduce, KleinSampler, Lattice, Sample, Vector};
use rug::{Float, Integer};

/**

Implements the Gass Sieve described in \[MV10\]

# Examples

```rust
use svp::*;
use rug::{Float, Integer};

// Initialize basis B
let mut b = vec![
    nvec![Integer::from(1), Integer::new(), Integer::new()],
    nvec![Integer::new(), Integer::from(1), Integer::new()],
    nvec![Integer::new(), Integer::new(), Integer::from(1)],
];

// Compute squared norm
for i in 0..b.len() {
    b[i].norm = Some(&b[i] * &b[i]);
}

// Init L(B)
let l = Lattice { basis: b.clone() };

// Rejection sampling parameter
let t = Float::with_val(32, b.len()).ln();


// Init Gauss Sieve
let mut gs = gsieve![l, t];

// Start Sieve
let short_vecs = gs.sieve();

// Short vectors sorted in ascending order
assert_eq!(short_vecs[0].norm, b[0].norm);
```
**/

#[derive(Debug)]
/// `GaussSieve` implements the sieving algorithm described in \[MV10\]
pub struct GaussSieve<T, U> {
    pub b: Lattice<T>, // LLL/BKZ reduced lattice basis
    pub k: KleinSampler<U>,
    pub l: Vec<Vector<T>>,
    pub s: Vec<Vector<T>>,
}

/// Mutually reduce sample list with respect to v
trait ListReduce<T> {
    /// After Gauss reduction, the angle between any
    /// two vectors in the list is at least 60 degrees
    fn reduce(&mut self, v: &mut Vector<T>);
}

/// Main `Sieve` loop
pub trait Sieve<T> {
    /// Returns a list of short vectors sorted in ascending order
    fn sieve(&mut self) -> Vec<Vector<T>>;
}

macro_rules! lr_impl {
    ($t:ty, $u:ty) => {
        impl ListReduce<$t> for GaussSieve<$t, $u> {
            fn reduce(&mut self, v: &mut Vector<$t>) {
                let mut index = 0;
                let mut reduced = true;
                while reduced {
                    reduced = false;
                    for i in 0..self.l.len() {
                        if self.l[i].norm > v.norm {
                            index = i;
                            break;
                        }
                        if v.reduce(&self.l[i]) {
                            reduced = true;
                        }
                    }
                }

                if v.norm.as_ref().unwrap() != &0 {
                    self.l.insert(index, v.clone());
                    index += 1;
                    while index < self.l.len() {
                        if self.l[index].reduce(&v) {
                            self.s.push(self.l[index].clone());
                            self.l.remove(index);
                        } else {
                            index += 1;
                        }
                    }
                }
            }
        }
    };
}

macro_rules! sieve_impl {
    ($t:ty, $u:ty) => {
        impl Sieve<$t> for GaussSieve<$t, $u> {
            fn sieve(&mut self) -> Vec<Vector<$t>> {
                let mut c = 0.0;
                let mut ml = self.l.len() as f64;
                let mut min_norm = self.b.basis[0].norm.clone();
                while c < ml * 0.1 + 200.0 {
                    let mut v: Vector<$t> = match self.s.is_empty() {
                        false => self.s.pop().unwrap(),
                        true => self.k.sample(&self.b),
                    };
                    self.reduce(&mut v);
                    if v.norm.as_ref().unwrap() == &0 {
                        c += 1.0;
                    } else if v.norm < min_norm {
                        min_norm = v.norm;
                    }
                    if self.l.len() as f64 > ml {
                        ml = self.l.len() as f64;
                    }
                }
                let mut res: Vec<Vector<$t>> = self.l.iter().map(|i| &self.b * i).collect();
                res.sort_by(|a, b| a.norm.partial_cmp(&b.norm).unwrap());
                res
            }
        }
    };
}

/**
Initializes the Gauss Sieve

# Examples

```rust
use svp::*;

// 3x3 Identity matrix
let mut b = vec![nvec![1, 0, 0], nvec![0, 1, 0], nvec![0, 0, 1]];

// Compute norms
for i in 0..b.len() {
    b[i].norm = Some(&b[i] * &b[i]);
}

// Init lattice
let l = Lattice { basis: b.clone() };

// Rejection sampling parameter
let t = (b.len() as f64).ln();

// Construct Gauss Sieve
let mut gs = gsieve![l, t];

// Run the sieve
let short_vecs = gs.sieve();

// Short vectors sorted in ascending order
assert_eq!(short_vecs[0].norm, b[0].norm);
```
**/

#[macro_export]
macro_rules! gsieve {
    ($l:expr,$t:expr) => {{
        GaussSieve {
            s: $l.basis.clone(),
            k: KleinSampler::init(&$l.gso(), $t),
            b: $l,
            l: vec![],
        }
    }};
}

/* Sieving type definitions */
lr_impl!(i64, f64);
sieve_impl!(i64, f64);
lr_impl!(Integer, Float);
sieve_impl!(Integer, Float);

#[cfg(test)]
mod tests {
    use crate::*;
    use rug::{Float, Integer};

    #[test]
    fn test_identity() {
        let mut b = vec![nvec![1, 0, 0], nvec![0, 1, 0], nvec![0, 0, 1]];
        for i in 0..b.len() {
            b[i].norm = Some(&b[i] * &b[i]);
        }

        let l = Lattice { basis: b.clone() };
        let t = (b.len() as f64).ln();

        let mut gs = gsieve![l, t];

        let short_vecs = gs.sieve();
        assert_eq!(short_vecs[0].norm, b[0].norm);
    }

    #[test]
    fn test_dim10() {
        let mut b = vec![
            nvec![-1, 0, 1, 0, 1, 0, 0, 0, -1, 1],
            nvec![-2, 2, -1, 0, 2, 3, 0, 1, 0, -2],
            nvec![-3, 1, -1, 1, 0, -4, -1, -2, 0, 0],
            nvec![1, 6, 0, 0, 1, 0, 2, 0, 0, 2],
            nvec![-2, 1, -4, -1, -1, 0, 0, 4, -3, 2],
            nvec![1, 0, -5, -10, 4, -3, -2, 0, 3, 4],
            nvec![5, 0, -4, 4, 6, -6, 0, 4, -9, -7],
            nvec![4, 3, -2, -7, -2, 3, 0, -6, -12, -2],
            nvec![1, 6, 0, 1, -3, 3, -15, 3, -1, 2],
            nvec![0, 3, 11, -9, -5, -4, -3, 8, -1, -7],
        ];

        for i in 0..b.len() {
            b[i].norm = Some(&b[i] * &b[i]);
        }

        let l = Lattice { basis: b.clone() };
        let t = (b.len() as f64).ln();

        let mut gs = gsieve![l, t];

        let short_vecs = gs.sieve();
        assert_eq!(short_vecs[0].norm.unwrap(), 62);
    }

    #[test]
    fn test_identity_mp() {
        let mut b = vec![
            nvec![Integer::from(1), Integer::new(), Integer::new()],
            nvec![Integer::new(), Integer::from(1), Integer::new()],
            nvec![Integer::new(), Integer::new(), Integer::from(1)],
        ];
        for i in 0..b.len() {
            b[i].norm = Some(&b[i] * &b[i]);
        }

        let l = Lattice { basis: b.clone() };
        let t = Float::with_val(32, b.len()).ln();

        let mut gs = gsieve![l, t];

        let short_vecs = gs.sieve();
        assert_eq!(short_vecs[0].norm, b[0].norm);
    }

    #[test]
    fn test_dim10_mp() {
        let mut b = vec![
            nvec![
                Integer::from(3),
                Integer::from(-4),
                Integer::from(-1),
                Integer::from(-2),
                Integer::new(),
                Integer::from(-2),
                Integer::from(-1),
                Integer::from(1),
                Integer::from(-1),
                Integer::new()
            ],
            nvec![
                Integer::from(-1),
                Integer::from(1),
                Integer::from(-6),
                Integer::from(1),
                Integer::from(-1),
                Integer::from(-2),
                Integer::from(-3),
                Integer::from(-1),
                Integer::from(3),
                Integer::new()
            ],
            nvec![
                Integer::from(1),
                Integer::new(),
                Integer::from(-7),
                Integer::from(3),
                Integer::new(),
                Integer::new(),
                Integer::from(1),
                Integer::from(1),
                Integer::from(-3),
                Integer::from(-1)
            ],
            nvec![
                Integer::new(),
                Integer::from(-2),
                Integer::from(1),
                Integer::from(5),
                Integer::new(),
                Integer::from(-3),
                Integer::from(-6),
                Integer::from(-3),
                Integer::from(-4),
                Integer::from(-1)
            ],
            nvec![
                Integer::from(-8),
                Integer::from(-1),
                Integer::from(-1),
                Integer::from(1),
                Integer::new(),
                Integer::from(-4),
                Integer::from(-2),
                Integer::from(1),
                Integer::new(),
                Integer::from(7)
            ],
            nvec![
                Integer::from(-7),
                Integer::from(-2),
                Integer::from(-1),
                Integer::from(-1),
                Integer::from(1),
                Integer::from(3),
                Integer::from(1),
                Integer::from(-1),
                Integer::from(-1),
                Integer::from(-9)
            ],
            nvec![
                Integer::from(3),
                Integer::from(5),
                Integer::from(3),
                Integer::from(5),
                Integer::from(-6),
                Integer::from(-11),
                Integer::from(-1),
                Integer::from(12),
                Integer::from(-4),
                Integer::from(-3)
            ],
            nvec![
                Integer::from(-1),
                Integer::from(9),
                Integer::from(-3),
                Integer::from(-13),
                Integer::from(-7),
                Integer::from(-3),
                Integer::from(-8),
                Integer::from(-6),
                Integer::from(-12),
                Integer::from(-2)
            ],
            nvec![
                Integer::from(2),
                Integer::from(-8),
                Integer::new(),
                Integer::from(12),
                Integer::from(-27),
                Integer::from(2),
                Integer::from(9),
                Integer::new(),
                Integer::from(11),
                Integer::from(1)
            ],
            nvec![
                Integer::from(-4),
                Integer::from(4),
                Integer::from(2),
                Integer::from(1),
                Integer::from(-2),
                Integer::from(-29),
                Integer::from(22),
                Integer::from(-17),
                Integer::from(-1),
                Integer::from(-6)
            ],
        ];

        for i in 0..b.len() {
            b[i].norm = Some(&b[i] * &b[i]);
        }

        let l = Lattice { basis: b.clone() };
        let t = Float::with_val(32, b.len()).ln();

        let mut gs = gsieve![l, t];

        let short_vecs = gs.sieve();
        assert_eq!(short_vecs[0].norm.as_ref().unwrap(), &111);
    }
}
