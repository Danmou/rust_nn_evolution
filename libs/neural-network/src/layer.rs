use crate::*;

use neuron::Neuron;

#[derive(Clone, Debug)]
pub(crate) struct Layer {
    pub(crate) neurons: Vec<Neuron>,
}

impl Layer {
    pub fn random(rng: &mut dyn rand::RngCore, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            neurons: (0..num_outputs)
                .map(|_| Neuron::random(rng, num_inputs))
                .collect(),
        }
    }

    pub fn propagate(&self, inputs: &[f32]) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
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
            let layer = Layer::random(&mut rng, 3, 2);

            let actual_biases: Vec<_> = layer.neurons.iter().map(|neuron| neuron.bias).collect();
            let expected_biases = vec![-0.6255188, 0.5238807];

            let actual_weights: Vec<_> = layer
                .neurons
                .iter()
                .map(|neuron| neuron.weights.as_slice())
                .collect();
            let expected_weights: Vec<&[f32]> = vec![
                &[0.67383957, 0.8181262, 0.26284897],
                &[-0.53516835, 0.069369674, -0.7648182],
            ];

            approx::assert_relative_eq!(actual_biases.as_slice(), expected_biases.as_slice());
            approx::assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let neurons = vec![
                Neuron {
                    bias: -0.1,
                    weights: vec![-0.1, 0.2, -0.3],
                },
                Neuron {
                    bias: 0.2,
                    weights: vec![0.4, -0.5, 0.6],
                },
            ];
            let layer = Layer { neurons: neurons.clone() };

            let inputs = &[-0.5, 0.0, 0.5];

            let actual = layer.propagate(inputs);
            let expected: Vec<_> = neurons
                .iter()
                .map(|neuron| neuron.propagate(inputs))
                .collect();

            approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
        }
    }
}
