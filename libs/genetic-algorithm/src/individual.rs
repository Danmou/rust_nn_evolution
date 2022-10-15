use crate::chromosome::Chromosome;

pub trait Individual {
    fn create(chromosome: Chromosome) -> Self;

    fn fitness(&self) -> f32;

    fn chromosome(&self) -> &Chromosome;
}
