use self::crossover_methods::CrossoverMethod;
use self::individual::Individual;
use self::mutation_methods::MutationMethod;
use self::selection_methods::SelectionMethod;
use rand::RngCore;

mod chromosome;
mod crossover_methods;
mod individual;
mod mutation_methods;
mod selection_methods;

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        (0..population.len())
            .map(|_| {
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();

                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);

                self.mutation_method.mutate(rng, &mut child);

                I::create(child)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chromosome::Chromosome;
    use crate::crossover_methods::UniformCrossover;
    use crate::mutation_methods::GaussianMutation;
    use crate::selection_methods::RouletteWheelSelection;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[derive(Clone, Debug, PartialEq)]
    pub struct TestIndividual {
        chromosome: Chromosome,
    }

    impl Individual for TestIndividual {
        fn create(chromosome: Chromosome) -> Self {
            Self { chromosome }
        }

        fn fitness(&self) -> f32 {
            // the simplest fitness function ever - we're just
            // summing all the genes together
            self.chromosome.iter().sum()
        }

        fn chromosome(&self) -> &Chromosome {
            &self.chromosome
        }
    }

    fn individual(genes: &[f32]) -> TestIndividual {
        let chromosome = genes.iter().cloned().collect();

        TestIndividual::create(chromosome)
    }

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5, 0.5),
        );

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]),  // fitness = 0.0
            individual(&[1.0, 1.0, 1.0]),  // fitness = 3.0
            individual(&[1.0, 2.0, 1.0]),  // fitness = 4.0
            individual(&[1.0, 2.0, 4.0]),  // fitness = 7.0
        ];

        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population);
        }

        let expected_population = vec![
            individual(&[1.5474501, 3.0097125, 5.058356]),  // fitness ~= 9.6
            individual(&[1.7977655, 2.7836556, 4.976598]),  // fitness ~= 9.6
            individual(&[2.137654, 3.0554078, 5.3231034]),  // fitness ~= 10.5
            individual(&[1.7108417, 3.0097125, 5.058356]),  // fitness ~= 9.8
        ];

        assert_eq!(population, expected_population);
    }
}
