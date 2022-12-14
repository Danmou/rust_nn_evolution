use crate::*;

#[derive(Clone, Debug)]
pub(crate) struct Neuron {
    pub(crate) bias: f32,
    pub(crate) weights: Vec<f32>,
}

impl Neuron {
    pub fn random(rng: &mut dyn rand::RngCore, num_inputs: usize) -> Self {
        Self {
            bias: rng.gen_range(-1.0..=1.0),
            weights: (0..num_inputs).map(|_| rng.gen_range(-1.0..=1.0)).collect(),
        }
    }

    pub fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());
        inputs
            .iter()
            .zip(&self.weights)
            .fold(self.bias, |sum, (input, weight)| sum + input * weight)
            .max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);

            approx::assert_relative_eq!(neuron.bias, -0.6255188);
            approx::assert_relative_eq!(
                neuron.weights.as_slice(),
                &[0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
            );
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            approx::assert_relative_eq!(
                neuron.propagate(&[-10.0, -10.0]),
                0.0,
            );

            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
            )
        }
    }
}
