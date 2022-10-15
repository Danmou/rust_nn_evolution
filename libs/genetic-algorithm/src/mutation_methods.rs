use crate::chromosome::Chromosome;
use rand::{Rng, RngCore};

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    /// Probability of changing a gene
    chance: f32,

    /// Magnitude of that change (std dev)
    coeff: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coeff: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);

        Self { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        child.iter_mut().for_each(|gene| {
            if rng.gen_bool(self.chance as _) {
                *gene += self.coeff * rng.gen::<f32>();
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const ORIGINAL: &'static [f32] = &[1.0, 2.0, 3.0, 4.0, 5.0];

    fn actual(chance: f32, coeff: f32) -> Vec<f32> {
        let mut child = ORIGINAL.iter().cloned().collect();

        let mut rng = ChaCha8Rng::from_seed(Default::default());

        GaussianMutation::new(chance, coeff).mutate(&mut rng, &mut child);

        child.into_iter().collect()
    }

    mod given_zero_chance {
        use super::ORIGINAL;

        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(0.0, coeff)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);

                approx::assert_relative_eq!(actual.as_slice(), ORIGINAL);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.5);

                approx::assert_relative_eq!(actual.as_slice(), ORIGINAL);
            }
        }
    }

    mod given_fifty_fifty_chance {
        use super::ORIGINAL;

        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(0.5, coeff)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);

                approx::assert_relative_eq!(actual.as_slice(), ORIGINAL);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn slightly_changes_the_original_chromosome() {
                let actual = actual(0.5);
                let expected = vec![1.0, 2.0, 3.2673423, 4.127801, 5.3188653];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }
    }

    mod given_max_chance {
        use super::ORIGINAL;

        fn actual(coeff: f32) -> Vec<f32> {
            super::actual(1.0, coeff)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn does_not_change_the_original_chromosome() {
                let actual = actual(0.0);

                approx::assert_relative_eq!(actual.as_slice(), ORIGINAL);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn entirely_changes_the_original_chromosome() {
                let actual = actual(0.5);
                let expected = vec![1.0936203, 2.41846, 3.4545314, 4.3157125, 5.38097];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }
    }
}
