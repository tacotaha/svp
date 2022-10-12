use crate::{nvec, Vector};
use rug::{Float, Integer};

const DEFAULT_PRECISION: u32 = 128;

/**
An Integer lattice generated by a full rank basis

# Examples

```rust
use svp::{nvec, Vector, Lattice, GSO};

// Defines a lattice generated by B
let l = Lattice {
    basis: vec![nvec![1, 1, 0], nvec![1, 2, 0], nvec![0, 1, 2]],
};

// Computes the Gram-Schmidt orthogonalization of B
let gs : Vec<Vector<f64>> = l.gso();

// Right multiplication by an n-vector corresponds to the matrix product
let lattice_point: Vector<i64> = &l * &nvec![1, 0, 0];
```

Integer Lattices with arbitrary precision

```rust
use rug::{Integer, Float};
use svp::{nvec, KleinSampler, Lattice, Sample, Vector, GSO};

// Defines a lattice generated by B
let l = Lattice {
    basis: vec![
        nvec![Integer::from(1), Integer::from(-1), Integer::from(1)],
        nvec![Integer::from(1), Integer::new(), Integer::from(1)],
        nvec![Integer::from(1), Integer::from(1), Integer::from(2)],
    ],
};

// Computes the Gram-Schmidt orthogonalization of B
let gs: Vec<Vector<Float>> = l.gso();

// Right multiplication by an n-vector corresponds to the matrix product
let lattice_point: Vector<Integer> = &l * &nvec![Integer::from(1), Integer::new(), Integer::new()];
```
**/

#[derive(Debug)]
/// A `Lattice` is generated by an nxm basis
pub struct Lattice<T> {
    pub basis: Vec<Vector<T>>,
}

/// Compute the Gram-Schmidt Orthogonalization of B
pub trait GSO<T> {
    fn gso(&self) -> Vec<Vector<T>>;
}

impl GSO<f64> for Lattice<i64> {
    fn gso(&self) -> Vec<Vector<f64>> {
        let n = self.basis.len();
        let m = self.basis[0].vec.len();
        let mut mu = vec![vec![0f64; m]; n];
        let mut gs: Vec<Vector<f64>> = vec![];

        for i in 0..self.basis.len() {
            let x: Vec<f64> = self.basis[i].vec.iter().map(|i| *i as f64).collect();
            gs.push(Vector {
                vec: x,
                norm: self.basis[i].norm.map(|z| z as f64),
            });
        }

        for i in 0..self.basis.len() {
            for j in 0..i {
                mu[i][j] = (&self.basis[j] * &gs[i]) / gs[j].norm.unwrap();
                for k in 0..self.basis.len() {
                    gs[i].vec[k] -= mu[i][j] * gs[j].vec[k];
                }
            }
            gs[i].norm = Some(&gs[i] * &gs[i]);
        }

        gs
    }
}

impl GSO<Float> for Lattice<Integer> {
    fn gso(&self) -> Vec<Vector<Float>> {
        let n = self.basis.len();
        let m = self.basis[0].vec.len();
        let mut mu = vec![vec![Float::new(DEFAULT_PRECISION); m]; n];
        let mut gs: Vec<Vector<Float>> = vec![];

        for i in 0..self.basis.len() {
            let x: Vec<Float> = self.basis[i]
                .vec
                .iter()
                .map(|i| Float::with_val(DEFAULT_PRECISION, i))
                .collect();
            gs.push(Vector {
                vec: x,
                norm: self.basis[i]
                    .norm
                    .as_ref()
                    .map(|z| Float::with_val(DEFAULT_PRECISION, z)),
            });
        }

        for i in 0..self.basis.len() {
            for j in 0..i {
                mu[i][j] = (&self.basis[j] * &gs[i]) / gs[j].norm.as_ref().unwrap();
                for k in 0..self.basis.len() {
                    let tmp = Float::with_val(DEFAULT_PRECISION, &mu[i][j] * &gs[j].vec[k]);
                    gs[i].vec[k] -= tmp;
                }
            }
            gs[i].norm = Some(&gs[i] * &gs[i]);
        }

        gs
    }
}

/// Right multiply basis matrix by a vector
impl std::ops::Mul<&Vector<f64>> for &Lattice<i64> {
    /// The resulting vector type of the matrix product
    type Output = Vector<i64>;
    /// Compute the matrix product with v
    fn mul(self, _rhs: &Vector<f64>) -> Vector<i64> {
        assert_eq!(self.basis.len(), _rhs.vec.len());
        let mut res = nvec![0i64; _rhs.vec.len()];
        for i in 0..self.basis.len() {
            res.vec[i] = (&self.basis[i] * _rhs) as i64;
        }
        res.norm = Some(&res * &res);
        res
    }
}

/// Right multiply basis matrix by a vector
impl std::ops::Mul<&Vector<i64>> for &Lattice<i64> {
    /// The resulting vector type of the matrix product
    type Output = Vector<i64>;
    /// Compute the matrix product with v
    fn mul(self, _rhs: &Vector<i64>) -> Vector<i64> {
        assert_eq!(self.basis.len(), _rhs.vec.len());
        let mut res = nvec![0i64; _rhs.vec.len()];
        for i in 0..self.basis.len() {
            res.vec[i] = &self.basis[i] * _rhs;
        }
        res.norm = Some(&res * &res);
        res
    }
}

/// Right multiply basis matrix by a vector with arbitrary precision
impl std::ops::Mul<&Vector<Float>> for &Lattice<Integer> {
    /// The resulting vector type of the matrix product
    type Output = Vector<Integer>;
    /// Compute the matrix product with v
    fn mul(self, _rhs: &Vector<Float>) -> Vector<Integer> {
        assert_eq!(self.basis.len(), _rhs.vec.len());
        let mut res = nvec![Integer::new(); _rhs.vec.len()];
        for i in 0..self.basis.len() {
            res.vec[i] = (&self.basis[i] * _rhs).to_integer().unwrap();
        }
        res.norm = Some(&res * &res);
        res
    }
}

/// Right multiply basis matrix by a vector with arbitrary precision
impl std::ops::Mul<&Vector<Integer>> for &Lattice<Integer> {
    /// The resulting vector type of the matrix product
    type Output = Vector<Integer>;
    /// Compute the matrix product with v
    fn mul(self, _rhs: &Vector<Integer>) -> Vector<Integer> {
        assert_eq!(self.basis.len(), _rhs.vec.len());
        let mut res = nvec![Integer::new(); _rhs.vec.len()];
        for i in 0..self.basis.len() {
            res.vec[i] = &self.basis[i] * _rhs;
        }
        res.norm = Some(&res * &res);
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use rug::Integer;

    #[test]
    fn test_prim() {
        let l = Lattice {
            basis: vec![nvec![1, 1, 0], nvec![1, 2, 0], nvec![0, 1, 2]],
        };
        let gs = l.gso();
        let ns: f64 = gs.iter().map(|i| i.norm.unwrap()).sum();
        assert_eq!(ns, 6.5);

        let l = Lattice {
            basis: vec![nvec![1, -1, 1], nvec![1, 0, 1], nvec![1, 1, 2]],
        };
        let gs = l.gso();
        let ns: f64 = gs.iter().map(|i| i.norm.unwrap()).sum();
        assert_eq!(ns.round(), 4.0);
    }

    #[test]
    fn test_mp() {
        let mut l = Lattice {
            basis: vec![
                nvec![Integer::from(1), Integer::new(), Integer::new()],
                nvec![Integer::new(), Integer::from(1), Integer::new()],
                nvec![Integer::new(), Integer::new(), Integer::from(1)],
            ],
        };

        for i in 0..l.basis.len() {
            l.basis[i].norm = Some(&l.basis[i] * &l.basis[i]);
        }

        let gs = l.gso();
        for i in 0..gs.len() {
            for j in 0..gs[i].vec.len() {
                assert_eq!(gs[i].vec[j], l.basis[i].vec[j]);
            }
        }

        let mut l = Lattice {
            basis: vec![
                nvec![Integer::from(1), Integer::from(-1), Integer::from(1)],
                nvec![Integer::from(1), Integer::new(), Integer::from(1)],
                nvec![Integer::from(1), Integer::from(1), Integer::from(2)],
            ],
        };

        for i in 0..l.basis.len() {
            l.basis[i].norm = Some(&l.basis[i] * &l.basis[i]);
        }

        let gs = l.gso();
        let mut sum = gs[0].norm.as_ref().unwrap().clone();
        for i in 1..gs.len() {
            sum += gs[i].norm.as_ref().unwrap();
        }
        assert_eq!(sum.round(), 4);
    }
}