#![feature(array_windows)]

use rand::Rng;

use self::layer::*;

mod layer;
mod neuron;

pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Clone, Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random(rng: &mut dyn rand::RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);
        Self {
            layers: layers
                .array_windows()
                .map(|[input, output]| Layer::random(rng, input.neurons, output.neurons))
                .collect(),
        }
    }

    pub fn propagate(&self, inputs: &[f32]) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs.to_vec(), |inputs, layer| layer.propagate(&inputs))
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
            let network = Network::random(
                &mut rng,
                &[
                    LayerTopology { neurons: 3 },
                    LayerTopology { neurons: 2 },
                    LayerTopology { neurons: 1 },
                ],
            );

            assert_eq!(network.layers.len(), 2);
            assert_eq!(network.layers[0].neurons.len(), 2);
            assert_eq!(network.layers[0].neurons[0].weights.len(), 3);
            assert_eq!(network.layers[0].neurons[1].weights.len(), 3);
            assert_eq!(network.layers[1].neurons.len(), 1);
            assert_eq!(network.layers[1].neurons[0].weights.len(), 2);
        }
    }

    mod propagate {
        use super::*;
        use crate::neuron::Neuron;

        #[test]
        fn test() {
            let layers = vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            bias: -0.3,
                            weights: vec![-0.5, 0.4, -0.3],
                        },
                        Neuron {
                            bias: 0.1,
                            weights: vec![-0.2, 0.1, 0.0],
                        },
                    ]
                },
                Layer {
                    neurons: vec![
                        Neuron {
                            bias: 0.2,
                            weights: vec![-0.5, 0.5],
                        },
                    ]
                },
            ];
            let network = Network {
                layers: layers.clone(),
            };

            let inputs = &[-0.5, 0.6, -0.7];

            let actual = network.propagate(inputs);
            let expected = layers[1].propagate(&layers[0].propagate(inputs));

            approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
        }
    }
}
