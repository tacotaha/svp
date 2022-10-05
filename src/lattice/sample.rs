use crate::{Lattice, Vector};
use rand::Rng;

#[derive(Debug)]
/// `KleinSampler` implements the sampling algorithm described in [GPV08]
pub struct KleinSampler {
    gs: Vec<Vector<f64>>, // Gram-Schmidt matrix
    t: f64,               // rejection sampling parameter
    s2: Vec<f64>,
}

impl KleinSampler {
    /// Initialize the `KleinSampler`
    pub fn init<T>(l: &Lattice<T>) -> Self
    where
        T: std::ops::Mul<T, Output = T>
            + std::ops::Add<T, Output = T>
            + std::default::Default
            + Copy,
        f64: From<T>,
    {
        let gs = l.gso();
        let n = gs.len();
        let t = (n as f64).ln();

        let mut max_norm = 0.0;
        for i in 0..n {
            if gs[i].norm > max_norm {
                max_norm = gs[i].norm;
            }
        }

        let s = max_norm * t;
        let mut s2 = vec![0f64; n];
        for i in 0..n {
            s2[i] = s / gs[i].norm;
        }

        Self { gs, t, s2 }
    }

    /// Rejection sample from the discrete gaussian
    fn sample_z(&self, c: f64, s2: f64) -> i32 {
        let s = s2.sqrt();
        let min = (c - s * self.t).floor();
        let max = (c + s * self.t).ceil();

        let mut rng = rand::thread_rng();
        loop {
            let deviate = rng.gen_range(0.0..1.0);
            let x = min + ((max - min) * deviate).round();
            let r = (-std::f64::consts::PI * (x - c) * (x - c) / s2).exp();
            if rng.gen_range(0.0..1.0) <= r {
                return x as i32;
            }
        }
    }

    /// The SampleD algorithm as described by GPV
    pub fn sample_d(&self) -> Vec<f64> {
        let mut coef = vec![0f64; self.gs.len()];

        for i in (0..coef.len()).rev() {
            coef[i] = self.sample_z(coef[i], self.s2[i]) as f64;
            for j in 0..i {
                coef[j] -= coef[i] * self.gs[i].vec[j];
            }
        }

        coef
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_z() {
        let b = vec![vec![1, 1, 0], vec![1, 2, 0], vec![0, 1, 2]];
        let l = Lattice::init(&b);
        let k = KleinSampler::init(&l);
        for _ in 0..100 {
            dbg!(k.sample_z(0.0, 6.0));
        }
    }

    #[test]
    fn test_sample_d() {
        let l = Lattice::init(&vec![vec![1, 1, 0], vec![1, 2, 0], vec![0, 1, 2]]);
        let k = KleinSampler::init(&l);
        for _ in 0..100 {
            let v = k.sample_d();
            dbg!(v);
        }
    }
}
