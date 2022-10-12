use rug::{Float, Integer};

/**

Generic n-vectors used to represent a basis

# Examples

```rust
use svp::{Vector, nvec};

// Build an integer vector
let u = Vector{ vec: vec![0i64; 3], norm: None};
let v = Vector{ vec: vec![0, 1, 0], norm: None};

// Shorthand notation
let u = nvec![0i64; 3];
let v = nvec![0, 1, 0];
```

Vector multiplication corresponds to the scalar product

```rust
use svp::{Vector, nvec};

let u = nvec![1, 0, 0];
let v = nvec![0, 1, 0];

assert_eq!(&u * &v, 0);
```

It can be useful to keep track of the squared norm

```rust
use svp::{Vector, nvec};

// Build an integer vector
let mut u = nvec![1, 0, 0];

// Compute the squared norm (inner product)
u.norm = Some(&u * &u);

assert_eq!(u.norm.unwrap(), 1);
```

As a special case, big integers can be used for arbitrary precision

```rust
use rug::Integer;
use svp::{Vector, nvec};

// Big Integers with gmp
let p = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
let u = nvec![Integer::from_str_radix(p, 16).unwrap(); 3];

let v = nvec![Integer::new(), Integer::from(1), Integer::new()];
assert_eq!(&v * &v, Integer::from(1));
```
**/

#[derive(Debug, Clone)]
pub struct Vector<T> {
    pub vec: Vec<T>,     // n-vector
    pub norm: Option<T>, // squared norm
}

/// `GaussReduce` with respect to v
pub trait GaussReduce<T> {
    fn reduce(&mut self, v: &Vector<T>) -> bool;
}

impl std::ops::Mul for &Vector<i64> {
    /// The resulting scalar type of the inner product
    type Output = i64;
    /// Compute the inner product of two n-vectors
    #[inline]
    fn mul(self, _rhs: &Vector<i64>) -> i64 {
        assert!(!self.vec.is_empty() && self.vec.len() == _rhs.vec.len());
        let mut res = self.vec[0] * _rhs.vec[0];
        for i in 1..self.vec.len() {
            res += self.vec[i] * _rhs.vec[i];
        }
        res
    }
}

impl std::ops::Mul for &Vector<f64> {
    /// The resulting scalar type of the inner product
    type Output = f64;
    /// Compute the inner product of two n-vectors
    #[inline]
    fn mul(self, _rhs: &Vector<f64>) -> f64 {
        assert!(!self.vec.is_empty() && self.vec.len() == _rhs.vec.len());
        let mut res: f64 = self.vec[0] * _rhs.vec[0];
        for i in 1..self.vec.len() {
            res += self.vec[i] * _rhs.vec[i];
        }
        res
    }
}

impl std::ops::Mul<&Vector<f64>> for &Vector<i64> {
    /// The resulting scalar type of the inner product
    type Output = f64;
    /// Compute the (truncated) inner product of two n-vectors
    #[inline]
    fn mul(self, _rhs: &Vector<f64>) -> f64 {
        assert!(!self.vec.is_empty() && self.vec.len() == _rhs.vec.len());
        let mut res: f64 = self.vec[0] as f64 * _rhs.vec[0];
        for i in 1..self.vec.len() {
            res += self.vec[i] as f64 * _rhs.vec[i];
        }
        res
    }
}

impl std::ops::Mul for &Vector<Integer> {
    /// The resulting scalar type of the inner product
    type Output = Integer;
    /// Compute the inner product of two arbitrary precision n-vectors
    #[inline]
    fn mul(self, _rhs: &Vector<Integer>) -> Integer {
        assert!(!self.vec.is_empty() && self.vec.len() == _rhs.vec.len());
        let mut res: Integer = Integer::from(&self.vec[0] * &_rhs.vec[0]);
        for i in 1..self.vec.len() {
            res += &self.vec[i] * &_rhs.vec[i];
        }
        res
    }
}

impl std::ops::Mul for &Vector<Float> {
    /// The resulting scalar type of the inner product
    type Output = Float;
    /// Compute the inner product of two arbitrary precision n-vectors
    #[inline]
    fn mul(self, _rhs: &Vector<Float>) -> Float {
        assert!(!self.vec.is_empty() && self.vec.len() == _rhs.vec.len());
        let mut res: Float = Float::with_val(self.vec[0].prec(), &self.vec[0] * &_rhs.vec[0]);
        for i in 1..self.vec.len() {
            res += &self.vec[i] * &_rhs.vec[i];
        }
        res
    }
}

impl std::ops::Mul<&Vector<Float>> for &Vector<Integer> {
    /// The resulting scalar type of the inner product
    type Output = Float;
    /// Compute the inner product of two arbitrary precision n-vectors
    #[inline]
    fn mul(self, _rhs: &Vector<Float>) -> Float {
        assert!(!self.vec.is_empty() && self.vec.len() == _rhs.vec.len());
        let prec = _rhs.vec[0].prec();
        let mut res: Float = Float::with_val(prec, &_rhs.vec[0] * &self.vec[0]);
        for i in 1..self.vec.len() {
            res += Float::with_val(prec, &self.vec[i] * &_rhs.vec[i]);
        }
        res
    }
}

impl GaussReduce<i64> for Vector<i64> {
    /// `GaussReduce` with respect to v
    fn reduce(&mut self, v: &Vector<i64>) -> bool {
        let ip = &*self * v;
        if v.norm.unwrap() < (ip << 1).abs() {
            let q = (ip as f64 / v.norm.unwrap() as f64).round() as i64;
            for i in 0..self.vec.len() {
                self.vec[i] -= q * v.vec[i];
            }
            self.norm = Some(&*self * &*self);
            return true;
        }
        false
    }
}

impl GaussReduce<Integer> for Vector<Integer> {
    /// `GaussReduce` with respect to v
    fn reduce(&mut self, v: &Vector<Integer>) -> bool {
        let ip = &*self * v;
        let ip2: Integer = ip.clone() * 2;
        if v.norm.as_ref().unwrap() < &ip2.abs() {
            let (q, _) = ip.div_rem_round(v.norm.clone().unwrap());
            for i in 0..self.vec.len() {
                self.vec[i] -= &q * &v.vec[i];
            }
            self.norm = Some(&*self * &*self);
            return true;
        }
        false
    }
}

/**
Shorthand notation for declaring n-vectors

# Examples

```rust
use svp::{Vector, nvec};

// Build an integer vector
let u = Vector{ vec: vec![0i64; 3], norm: None};
let v = Vector{ vec: vec![0, 1, 0], norm: None};

// Shorthand notation
let u = nvec![0i64; 3];
let v = nvec![0, 1, 0];
```
**/
#[macro_export]
macro_rules! nvec {
    ($elem:expr; $n:expr) => (
        Vector {
            vec: vec![$elem; $n],
            norm: None,
        }
    );
    ($($x:expr),*) => (
        Vector {
            vec: <[_]>::into_vec(Box::new([$($x),*])),
            norm: None,
        }
    );
    ($($x:expr,)*) => (nvec![$($x),*])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prim() {
        let mut v1 = nvec![1; 5];
        v1.norm = Some(&v1 * &v1);
        assert_eq!(v1.norm.unwrap(), 5);

        let mut e0 = nvec![1, 0, 0];
        let mut e1 = nvec![0, 1, 0];
        let mut e2 = nvec![0, 0, 1];

        e0.norm = Some(&e0 * &e0);
        e1.norm = Some(&e0 * &e0);
        e2.norm = Some(&e0 * &e0);
        assert!(e0.norm == e1.norm && e1.norm == e2.norm);
    }

    #[test]
    fn test_mp() {
        let mut v1 = nvec![Integer::from(1); 5];
        v1.norm = Some(&v1 * &v1);
        assert_eq!(v1.norm.unwrap(), 5);

        let mut e0 = nvec![Integer::from(1), Integer::new(), Integer::new()];
        let mut e1 = nvec![Integer::new(), Integer::from(1), Integer::new()];
        let mut e2 = nvec![Integer::new(), Integer::new(), Integer::from(1)];

        e0.norm = Some(&e0 * &e0);
        e1.norm = Some(&e1 * &e1);
        e2.norm = Some(&e2 * &e2);
        assert!(e0.norm == e1.norm && e1.norm == e2.norm);
    }
}
