use tch::nn::ModuleT;
use tch::{nn, IndexOp, Tensor};

use crate::model::{Block, BlockConfig, LanguageModel, LayerNorm, LayerNormConfig};

/// NanoGpt is a language model.
#[derive(Debug)]
pub struct NanoGpt {
    /// The embedding layer
    token_embedding: nn::Embedding,
    /// The position embedding layer
    position_embedding: nn::Embedding,
    /// LM head
    lm_head: nn::Linear,
    /// The embedding size
    n_embd: i64,
    /// The vocabulary size
    vocab_size: i64,
    /// The maximum sequence length
    block_size: usize,
    /// Layers
    layers: nn::SequentialT,
    /// Layer normalization
    ln_f: LayerNorm,
    /// Dropout probability
    dropout: f64,
}

/// NanoGpt configuration.
#[derive(Debug, Clone)]
pub struct NanoGptConfig {
    /// The vocabulary size.
    pub vocab_size: i64,
    /// The maximum sequence length.
    pub block_size: usize,
    /// The embedding size.
    pub n_embd: i64,
    /// The number of heads.
    pub n_head: i64,
    /// The number of layers.
    pub n_layer: i64,
    /// The dropout probability.
    pub dropout: f64,
    /// Biases flag.
    pub bias: bool,
    /// Weight tying for WTE and LM head.
    pub tie_weights: bool,
}

impl NanoGpt {
    /// Create a new NanoGpt
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `config` - The model configuration. See [NanoGptConfig].
    /// # Returns
    /// A new NanoGpt.
    pub fn new(vs: &nn::Path, config: NanoGptConfig) -> Self {
        let block_config = BlockConfig::from(&config);

        let NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_layer,
            dropout,
            ..
        } = config;

        let token_embedding = nn::embedding(vs / "wte", vocab_size, n_embd, Default::default());

        let position_embedding =
            nn::embedding(vs / "wpe", block_size as i64, n_embd, Default::default());

        let mut layers = nn::seq_t();
        for i in 0..n_layer {
            layers = layers.add(Block::new(vs / "h" / i, block_config));
        }

        // lm_head with weight tying to token_embedding weight
        let lm_head = if config.tie_weights {
            nn::Linear {
                ws: token_embedding.ws.shallow_clone(),
                bs: None,
            }
        } else {
            nn::linear(
                vs / "lm_head",
                n_embd,
                vocab_size,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            )
        };

        let ln_f = LayerNorm::new(
            vs / "ln_f",
            vec![n_embd],
            LayerNormConfig {
                elementwise_bias: config.bias,
                ..Default::default()
            },
        );

        Self {
            token_embedding,
            position_embedding,
            lm_head,
            layers,
            n_embd,
            vocab_size,
            block_size,
            ln_f,
            dropout,
        }
    }
}

impl nn::ModuleT for NanoGpt {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, t) = xs.size2().unwrap();

        // tok_emb
        let tok_emb = xs.apply(&self.token_embedding); // [batch_size, block_size, n_embd]

        // pos_emb
        let device = xs.device();
        let pos = Tensor::arange_start(0, t, (tch::Kind::Int, device)); // [t]
        let pos = pos.unsqueeze(0); // [1, t]
        assert_eq!(pos.size(), &[1, t]);

        let pos_emb = pos.apply(&self.position_embedding);
        let pos_emb = pos_emb.view([t, self.n_embd]); // [block_size, n_embd]
        assert_eq!(pos_emb.size(), &[t, self.n_embd]);

        // residual connection
        let x = tok_emb + pos_emb; // [batch_size, block_size, n_embd]
        assert_eq!(x.size(), &[b, t, self.n_embd]);

        let x = x.dropout(self.dropout, train); // [batch_size, block_size, n_embd]

        // layers
        let x = x.apply_t(&self.layers, train); // [batch_size, block_size, n_embd]

        // layer norm
        let x = x.apply_t(&self.ln_f, train); // [batch_size, block_size, n_embd]

        // lm head
        let x = x.apply_t(&self.lm_head, train); // [batch_size, block_size, vocab_size]
        assert_eq!(x.size(), &[b, t, self.vocab_size]);
        x
    }
}

impl LanguageModel for NanoGpt {
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
        let (b, _t) = xs.size2().unwrap();
        // assert_eq!(t, self.block_size);

        // Create a tensor of zeros to store the output
        let outputs = xs.new_zeros([b, max_len], (tch::Kind::Int64, xs.device()));

        let mut xs = xs;
        for i in 0..max_len {
            // take the last probabilities
            let probs = self.probabilities(&xs);
            // sample the next token
            let next_token = probs.multinomial(1, true);

            // add the next token to the output
            outputs.narrow(1, i, 1).copy_(&next_token);

            // update xs
            xs = Tensor::cat(&[xs, next_token], 1);

            // crop the sequence to the maximum length, starting from the end if needed
            // NOFUTURE(ssoudan) better way?
            let (_b, t) = xs.size2().unwrap();
            if t > self.block_size as i64 {
                xs = xs.i((.., (t - (self.block_size as i64))..));
            }
            assert!(xs.size()[1] <= self.block_size as i64);
        }

        outputs
    }

    fn probabilities(&self, xs: &Tensor) -> Tensor {
        let (b, _t) = xs.size2().unwrap();
        // take the last logits
        let logits = self.forward_t(xs, false).i((.., -1, ..));
        assert_eq!(logits.size(), &[b, self.vocab_size]);
        //  apply softmax to get the probabilities of the next token
        logits.softmax(-1, tch::Kind::Float).detach()
    }

    fn block_size(&self) -> usize {
        self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{loss, LanguageModel};

    /// Test the NanoGpt forward pass
    #[test]
    fn test_nano_gpt_forward() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vocab_size = 100;
        let batch_size = 2;
        let block_size = 5;
        let n_embd = 32;
        let n_head = 4;
        let n_layer = 2;
        let bias = true;
        let dropout = 0.1;
        let tie_weights = true;

        let config = NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
            bias,
            dropout,
            tie_weights,
        };

        let model = NanoGpt::new(&vs.root(), config);
        let xs = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size as i64, block_size as i64]);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size as i64);
        assert_eq!(t, block_size as i64);

        let logits = model.forward_t(&xs, false);
        assert_eq!(
            logits.size(),
            [batch_size as i64, block_size as i64, vocab_size]
        );

        let _loss = loss(&logits, &xs);
    }

    /// Test the NanoGpt forward pass with a sequence shorter than the maximum
    /// length (block_size)
    #[test]
    fn test_nano_gpt_forward_shorter_sequence() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vocab_size = 100;
        let batch_size = 2;
        let block_size = 5;
        let n_embd = 32;
        let n_head = 4;
        let n_layer = 2;
        let bias = true;
        let dropout = 0.1;
        let tie_weights = true;

        let config = NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
            bias,
            dropout,
            tie_weights,
        };

        let model = NanoGpt::new(&vs.root(), config);
        let xs = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size as i64 - 2]);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size as i64 - 2);

        let logits = model.forward_t(&xs, false);
        assert_eq!(
            logits.size(),
            [batch_size, block_size as i64 - 2, vocab_size]
        );

        let _loss = loss(&logits, &xs);
    }

    #[test]
    fn test_nano_gpt_generate() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vocab_size = 100;
        let batch_size = 2;
        let block_size = 5;
        let n_embd = 32;
        let n_head = 4;
        let n_layer = 2;
        let bias = true;
        let dropout = 0.1;
        let tie_weights = true;

        let config = NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
            bias,
            dropout,
            tie_weights,
        };

        let model = NanoGpt::new(&vs.root(), config);
        let xs = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size as i64]);
        println!("xs: {:?}", xs);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size as i64);

        let logits = model.forward_t(&xs, false);
        assert_eq!(logits.size(), [batch_size, block_size as i64, vocab_size]);

        let loss = loss(&logits, &xs);
        println!("loss: {:?}", loss);
        // we expect the loss to be close to -ln(1/vocab_size) = 4.17

        // use 0 as start of sequence token - this is '\n' with our data and tokenizer.
        let xs = Tensor::zeros([batch_size, 1], (tch::Kind::Int, tch::Device::Cpu));
        println!("xs: {:?}", xs);
        let max_len = 10;
        let ys = model.generate(xs, max_len);
        println!("generated: {:?}", ys);

        // decode the generated sequence of tokens
        let ys = ys.to_kind(tch::Kind::Int);

        println!("generated: {:?}", ys);

        let first = ys.i((0, ..));
        let first: Vec<i64> = first.reshape(-1).try_into().unwrap();
        println!("first: {:?}", first);

        let second = ys.i((1, ..));
        let second: Vec<i64> = second.reshape(-1).try_into().unwrap();
        println!("second: {:?}", second);
    }
}
