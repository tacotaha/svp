#[derive(Debug)]
/// Defines an n-`Vector`
pub struct Vector<T> {
    pub vec: Vec<T>, // n-vector
    pub norm: T,     // squared norm
}

impl<T> Vector<T>
where
    T: std::ops::Mul<T, Output = T> + std::ops::Add<T, Output = T> + std::default::Default + Copy,
{
    pub fn new(n: usize) -> Self {
        return Self {
            vec: vec![T::default(); n],
            norm: T::default(),
        };
    }

    /// Initialize an n-vector
    pub fn init(vec: &Vec<T>) -> Self {
        let mut v = Self {
            vec: vec.to_vec(),
            norm: T::default(),
        };
        v.norm = v.iprod(&v);
        v
    }

    /// Compute the inner product of two n-vectors
    pub fn iprod<U: Into<T> + Copy>(&self, v: &Vector<U>) -> T {
        assert!(self.vec.len() > 0 && self.vec.len() == v.vec.len());
        let mut res = self.vec[0] * v.vec[0].into();
        for i in 1..self.vec.len() {
            let p = self.vec[i] * v.vec[i].into();
            res = res + p;
        }
        res
    }

    pub fn to_f64(&self) -> Vector<f64>
    where
        f64: From<T>,
    {
        Vector {
            vec: self.vec.iter().map(|s| (*s).into()).collect(),
            norm: self.norm.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_mixed() {
        const N: i32 = 1 << 10;
        let mut rng = thread_rng();
        let dist = Uniform::new_inclusive(-N, N);

        for _ in 0..N {
            let dim: usize = rng.gen();
            let u: Vec<i32> = (&mut rng)
                .sample_iter(dist)
                .take(dim % (N as usize) + 1)
                .collect();

            let u_vec = Vector::init(&u);
            let unorm = u_vec.norm;

            let v_vec = u_vec.to_f64();
            let vnorm = v_vec.norm;

            let ip = v_vec.iprod(&u_vec);
            assert_eq!(unorm, vnorm as i32);
            assert_eq!(ip, vnorm);
        }
    }
}
