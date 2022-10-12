use crate::{nvec, Lattice, Vector};
use rand::Rng;
use rug::{Float, Integer};

/**

Implements the sampling algorithm described in \[GPV08\]

# Examples

```rust
use svp::{nvec, KleinSampler, Lattice, Sample, Vector, GSO};

// Initialize lattice
let l = Lattice {
    basis: vec![nvec![1, 1, 0], nvec![1, 2, 0], nvec![0, 1, 2]],
};

// Compute Gram-Schmidt matrix
let gs = l.gso();

// See sec 4.1 for definition of t(n)
let t = (gs.len() as f64).ln();

// Initialize the sampler
let k = KleinSampler::init(&gs, t);

// Sample lattice points
for _ in 0..10 {
    let s : Vector<i64> = k.sample(&l);
}

```

Sampling with arbitrary precision

```rust
use rug::{Integer, Float};
use svp::{nvec, KleinSampler, Lattice, Sample, Vector, GSO};

// Init lattice
let l = Lattice {
    basis: vec![
        nvec![Integer::from(1), Integer::from(-1), Integer::from(1)],
        nvec![Integer::from(1), Integer::new(), Integer::from(1)],
        nvec![Integer::from(1), Integer::from(1), Integer::from(2)],
    ],
};

// Compute Gram-Schmidt matrix
let gs = l.gso();

// Rejection sampling parameter
let t = Float::with_val(53, gs.len()).ln();

// Init sampler
let k = KleinSampler::init(&gs, t);

// Sample lattice points
for _ in 0..10 {
    let s : Vector<Integer> = k.sample(&l);
}
```
**/

#[derive(Debug)]
pub struct KleinSampler<T> {
    gs: Vec<Vector<T>>, // Gram-Schmidt matrix
    t: T,               // rejection sampling parameter
    s2: Vec<T>,
}

/// Rejection sample from the discrete gaussian
trait SampleZ<T> {
    fn sample_z(&self, c: &T, s: &T) -> T;
}

/// The SampleD subroutine as described in \[GPV08\]
pub trait Sample<T> {
    fn sample(&self, l: &Lattice<T>) -> Vector<T>;
}

impl<T> KleinSampler<T> {
    /// Initialize the `KleinSampler`
    pub fn init(gs: &Vec<Vector<T>>, t: T) -> Self
    where
        T: std::ops::Mul<T, Output = T> + std::ops::Div<T, Output = T> + Clone + PartialOrd,
    {
        let mut max_norm: T = gs[0].norm.clone().unwrap();
        for g in gs {
            if g.norm.clone().unwrap() > max_norm {
                max_norm = g.norm.clone().unwrap();
            }
        }

        let s = max_norm * t.clone();
        let s2 = gs
            .iter()
            .map(|i| s.clone() / i.norm.clone().unwrap())
            .collect();

        Self {
            gs: gs.to_vec(),
            t,
            s2,
        }
    }
}

impl SampleZ<f64> for KleinSampler<f64> {
    /// Rejection sample from the discrete gaussian
    fn sample_z(&self, c: &f64, s2: &f64) -> f64 {
        let s = s2.sqrt();
        let min = (c - s * self.t).floor();
        let max = (c + s * self.t).ceil();

        let mut rng = rand::thread_rng();
        loop {
            let deviate = rng.gen_range(0.0..1.0);
            let x = min + ((max - min) * deviate).round();
            let r = (-std::f64::consts::PI * (x - c) * (x - c) / s2).exp();
            if rng.gen_range(0.0..1.0) <= r {
                return x;
            }
        }
    }
}

impl Sample<i64> for KleinSampler<f64> {
    /// Sample a coefficient vector
    fn sample(&self, l: &Lattice<i64>) -> Vector<i64> {
        let mut coef = nvec![0f64; self.gs.len()];
        for i in (0..coef.vec.len()).rev() {
            coef.vec[i] = self.sample_z(&coef.vec[i], &self.s2[i]);
            for j in 0..i {
                coef.vec[j] -= coef.vec[i] * self.gs[i].vec[j];
            }
        }
        l * &coef
    }
}

impl SampleZ<Float> for KleinSampler<Float> {
    /// Rejection sample from the discrete gaussian with arbitrary precision
    fn sample_z(&self, c: &Float, s2: &Float) -> Float {
        let prec = c.prec();
        let s = s2.clone().sqrt();
        let min = Float::with_val(prec, c - &s * &self.t).floor();
        let max = Float::with_val(prec, c + &s * &self.t).ceil();
        let delta: Float = max - min.clone();
        let mut rng = rand::thread_rng();

        loop {
            let deviate = rng.gen_range(0.0..1.0);
            let x: Float = Float::with_val(prec, &delta * deviate).round() + &min;
            let u = Float::with_val(prec, &x - c).square();
            let r = (-std::f64::consts::PI * u / s2).exp();
            if rng.gen_range(0.0..1.0) <= r {
                return x;
            }
        }
    }
}

impl Sample<Integer> for KleinSampler<Float> {
    /// Sample a coefficient vector with arbitrary precision
    fn sample(&self, l: &Lattice<Integer>) -> Vector<Integer> {
        let prec = self.gs[0].vec[0].prec();
        let mut coef = nvec![Float::new(prec); self.gs.len()];
        for i in (0..coef.vec.len()).rev() {
            coef.vec[i] = Float::with_val(prec, self.sample_z(&coef.vec[i], &self.s2[i]));
            for j in 0..i {
                let tmp = Float::with_val(prec, &self.gs[i].vec[j] * &coef.vec[i]);
                coef.vec[j] -= tmp;
            }
        }
        l * &coef
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use rug::{Float, Integer};

    #[test]
    fn test_prim() {
        let l = Lattice {
            basis: vec![nvec![1, 1, 0], nvec![1, 2, 0], nvec![0, 1, 2]],
        };
        let gs = l.gso();
        let t = (gs.len() as f64).ln();
        let k = KleinSampler::init(&gs, t);
        for _ in 0..10 {
            assert_eq!(k.sample(&l).vec.len(), l.basis[0].vec.len());
        }
    }

    #[test]
    fn test_mp() {
        let l = Lattice {
            basis: vec![
                nvec![Integer::from(1), Integer::from(-1), Integer::from(1)],
                nvec![Integer::from(1), Integer::new(), Integer::from(1)],
                nvec![Integer::from(1), Integer::from(1), Integer::from(2)],
            ],
        };
        let gs = l.gso();
        let t = Float::with_val(53, gs.len()).ln();
        let k = KleinSampler::init(&gs, t);
        for _ in 0..10 {
            assert_eq!(k.sample(&l).vec.len(), l.basis[0].vec.len());
        }
    }
}
