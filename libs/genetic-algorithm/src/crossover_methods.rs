use crate::chromosome::Chromosome;
use rand::{Rng, RngCore};

pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome;
}

#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        parent_a
            .iter()
            .zip(parent_b.iter())
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod uniform_crossover {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            let method = UniformCrossover::new();
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let parent_a: Chromosome = (1..=100).map(|n| n as _).collect();
            let parent_b: Chromosome = (1..=100).map(|n| -n as _).collect();

            let child = method.crossover(&mut rng, &parent_a, &parent_b);

            let num_diff_a = child.iter().zip(parent_a.iter()).filter(|(&a, &b)| a != b).count();
            let num_diff_b = child.iter().zip(parent_b.iter()).filter(|(&a, &b)| a != b).count();

            assert_eq!(num_diff_a, 49);
            assert_eq!(num_diff_b, 51);
        }
    }
}
