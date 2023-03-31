use std::borrow::Borrow;

use tch::nn::{Init, LinearConfig, ModuleT, Path, SequentialT};
use tch::{nn, IndexOp, Tensor};

/// BigramLanguageModel is a language model that uses the current token to
/// predict the next token.
#[derive(Debug)]
pub struct BigramLanguageModel {
    /// The embedding layer
    embedding: nn::Embedding,
}

impl BigramLanguageModel {
    /// Create a new BigramLanguageModel
    pub fn new(vs: &Path, vocab_size: i64) -> Self {
        let embedding = nn::embedding(vs / "embedding", vocab_size, vocab_size, Default::default());

        Self { embedding }
    }

    /// Forward pass
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.embedding)
        // xs.apply(&self.linear)
    }
}

impl LMModel for BigramLanguageModel {
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
        let mut xs = xs;
        for _ in 0..max_len {
            // take the last logits
            let logits = self.forward(&xs).i((.., -1, ..));
            //  apply softmax to get the probabilities of the next token
            let probs = logits.softmax(-1, tch::Kind::Float);
            // sample the next token
            let next_token = probs.multinomial(1, true);
            xs = Tensor::cat(&[xs, next_token], 1);
        }
        xs
    }
}

impl ModuleT for BigramLanguageModel {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// One head self-attention.
#[derive(Debug)]
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    mask: Tensor,
    head_size: i64,
}

impl Head {
    /// Create a new Head.
    ///
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `block_size` - The maximum sequence length.
    /// * `n_emb` - The embedding size.
    /// * `head_size` - The size of the head.
    ///
    /// # Returns
    /// A new Head.
    ///
    /// # Notes
    /// The input of `forward` is expected to be of shape `[batch_size,
    /// block_size, C]`.
    pub fn new<'a, T: Borrow<Path<'a>>>(
        vs: T,
        block_size: i64,
        n_emb: i64,
        head_size: i64,
    ) -> Self {
        let vs = vs.borrow();

        let device = vs.device();

        let key = nn::linear(
            vs / "key",
            n_emb,
            head_size,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let query = nn::linear(
            vs / "query",
            n_emb,
            head_size,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let value = nn::linear(
            vs / "value",
            n_emb,
            head_size,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let mask = Tensor::ones(&[block_size, block_size], (tch::Kind::Float, device)).tril(0);
        Self {
            key,
            query,
            value,
            mask,
            head_size,
        }
    }
}

impl ModuleT for Head {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, t, _c) = xs.size3().unwrap();
        let head_size = self.head_size;

        // the dimension of the keys
        let d_k = head_size;

        let k = xs.apply(&self.key); // [b, t, head_size]
        assert_eq!(k.size(), &[b, t, head_size]);
        let q = xs.apply(&self.query); // [b, t, head_size]
        assert_eq!(q.size(), &[b, t, head_size]);
        let v = xs.apply(&self.value); // [b, t, head_size]
        assert_eq!(v.size(), &[b, t, head_size]);

        // Attention scores
        let wei = q.matmul(&k.transpose(-2, -1)) / (d_k as f64).sqrt();
        let wei = wei.masked_fill(&self.mask.i((..t, ..t)).eq(0.), f64::NEG_INFINITY);
        assert_eq!(wei.size(), &[b, t, t]);

        let wei = wei.softmax(-1, tch::Kind::Float);
        assert_eq!(wei.size(), &[b, t, t]);

        let wei = wei.dropout(0.2, train);

        // weighted aggregation of the values
        let out = wei.matmul(&v); // [b, t, head_size]
        assert_eq!(out.size(), &[b, t, head_size]);

        out
    }
}

/// Configuration for the MultiHeadSelfAttention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiHeadSelfAttentionConfig {
    /// The maximum sequence length.
    pub block_size: i64,
    /// The embedding size.
    pub n_embd: i64,
    /// The size of the head.
    pub head_size: i64,
    /// The number of heads.
    pub n_head: i64,
}

impl From<&BlockConfig> for MultiHeadSelfAttentionConfig {
    fn from(config: &BlockConfig) -> Self {
        Self {
            block_size: config.block_size,
            n_embd: config.n_embd,
            head_size: config.head_size,
            n_head: config.n_head,
        }
    }
}

/// MultiHeadSelfAttention is a multi-head self-attention layer.
#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    heads: Vec<Head>,
    projection: nn::Linear,
}

impl MultiHeadSelfAttention {
    /// Create a new MultiHeadSelfAttention.
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `config` - The configuration. See [`MultiHeadSelfAttentionConfig`].
    /// # Returns
    /// A new MultiHeadSelfAttention.
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, config: MultiHeadSelfAttentionConfig) -> Self {
        let MultiHeadSelfAttentionConfig {
            block_size,
            n_embd: n_emb,
            head_size,
            n_head,
        } = config;

        let vs = vs.borrow();

        let heads = (0..n_head)
            .map(|i| Head::new(vs / i, block_size, n_emb, head_size))
            .collect();

        let projection = nn::linear(
            vs / "projection",
            n_emb,
            n_emb,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self { heads, projection }
    }
}

impl ModuleT for MultiHeadSelfAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // concatenate the heads along the last dimension
        let heads = self
            .heads
            .iter()
            .map(|h| h.forward_t(xs, train))
            .collect::<Vec<_>>();

        let out = Tensor::cat(&heads, -1);

        out.apply(&self.projection).dropout(0.2, train)
    }
}

/// Feed forward layer
#[derive(Debug)]
struct FeedForward {
    net: SequentialT,
}

impl FeedForward {
    /// Create a new FeedForward
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, n_embd: i64) -> Self {
        let vs = vs.borrow();
        let net = nn::seq_t()
            .add(nn::linear(
                vs / "net",
                n_embd,
                4 * n_embd,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "projection",
                4 * n_embd,
                n_embd,
                Default::default(),
            ))
            .add_fn_t(|xs, train| xs.dropout(0.2, train));

        Self { net }
    }
}

impl ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.net.forward_t(xs, train)
    }
}

/// Block configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockConfig {
    /// The maximum sequence length.
    pub block_size: i64,
    /// The embedding size.
    pub n_embd: i64,
    /// The size of the head.
    pub head_size: i64,
    /// The number of heads.
    pub n_head: i64,
}

/// LayerNorm configuration.
#[derive(Debug, Clone, Copy)]
struct LayerNormConfig {
    /// Whether to use CUDNN.
    pub cudnn_enabled: bool,
    /// A small constant added to the denominator for numerical stability.
    pub eps: f64,
    /// Whether to apply a linear transformation.
    elementwise_linear: bool,
    /// Whether to apply a bias.
    elementwise_bias: bool,
    /// The weight initialization.
    ws_init: Init,
    /// The bias initialization.
    bs_init: Init,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            elementwise_linear: true,
            elementwise_bias: true,
            eps: 1e-5,
            cudnn_enabled: true,
            ws_init: Init::Const(1.),
            bs_init: Init::Const(0.),
        }
    }
}

/// Layer normalization with no biases.
///
/// - See [Layer Normalization](https://arxiv.org/abs/1607.06450).
/// - See [`nn::LayerNorm`].
#[derive(Debug)]
pub struct LayerNorm {
    /// The configuration.
    config: LayerNormConfig,
    /// The weight.
    pub ws: Option<Tensor>,
    /// The bias.
    pub bs: Option<Tensor>,
    /// The normalized shape.
    pub normalized_shape: Vec<i64>,
}

fn layer_norm<'a, T: Borrow<Path<'a>>>(
    vs: T,
    normalized_shape: Vec<i64>,
    config: LayerNormConfig,
) -> LayerNorm {
    let vs = vs.borrow();

    let ws = if config.elementwise_linear {
        let ws = vs.var("weight", normalized_shape.as_slice(), config.ws_init);
        Some(ws)
    } else {
        None
    };

    let bs = if config.elementwise_bias {
        let bs = vs.var("bias", normalized_shape.as_slice(), config.bs_init);
        Some(bs)
    } else {
        None
    };

    LayerNorm {
        config,
        ws,
        bs,
        normalized_shape,
    }
}

impl nn::Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::layer_norm(
            xs,
            self.normalized_shape.as_slice(),
            self.ws.as_ref(),
            self.bs.as_ref(),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}

impl From<&NanoGptConfig> for BlockConfig {
    fn from(config: &NanoGptConfig) -> Self {
        Self {
            block_size: config.block_size,
            n_embd: config.n_embd,
            head_size: config.n_embd / config.n_head,
            n_head: config.n_head,
        }
    }
}

/// A Block is a transformer decoder block.
#[derive(Debug)]
struct Block {
    /// SA heads
    sa_heads: MultiHeadSelfAttention,
    /// Feed forward layer
    ffwd: FeedForward,
    /// Layer normalization
    ln1: LayerNorm,
    /// Layer normalization
    ln2: LayerNorm,
}

impl Block {
    /// Create a new Block.
    ///
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `config` - The configuration. See [`BlockConfig`].
    ///
    /// # Returns
    /// A new Block.
    ///
    /// # Notes
    /// `n_embd` must be divisible by `n_head`.
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, config: BlockConfig) -> Self {
        let vs = vs.borrow();

        let mhsa_config = MultiHeadSelfAttentionConfig::from(&config);

        let BlockConfig { n_embd, n_head, .. } = config;

        assert_eq!(n_embd % n_head, 0, "n_emb must be divisible by n_head");

        let sa_heads = MultiHeadSelfAttention::new(vs / "sa_heads", mhsa_config);
        let ffwd = FeedForward::new(vs / "ffwd", n_embd);

        let ln1 = layer_norm(
            vs / "ln1",
            vec![n_embd],
            LayerNormConfig {
                elementwise_bias: false,
                ..Default::default()
            },
        );
        let ln2 = layer_norm(
            vs / "ln2",
            vec![n_embd],
            LayerNormConfig {
                elementwise_bias: false,
                ..Default::default()
            },
        );

        Self {
            sa_heads,
            ffwd,
            ln1,
            ln2,
        }
    }
}

impl ModuleT for Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // SA heads with residual connection
        let xs = xs + xs.apply_t(&self.ln1, train).apply_t(&self.sa_heads, train);
        // [b, t, n_embd]

        // Feed forward layer with residual connection
        &xs + &xs.apply_t(&self.ln2, train).apply_t(&self.ffwd, train)
        // [b, t, n_embd]
    }
}

/// NanoGpt is a language model.
#[derive(Debug)]
pub struct NanoGpt {
    /// The embedding layer
    token_embedding: nn::Embedding,
    /// The position embedding layer
    pub position_embedding: nn::Embedding,
    /// LM head
    lm_head: nn::Linear,
    /// The embedding size
    n_embd: i64,
    /// The vocabulary size
    vocab_size: i64,
    /// The maximum sequence length
    block_size: i64,
    /// Layers
    layers: SequentialT,
    /// Layer normalization
    ln: LayerNorm,
}

/// NanoGpt configuration.
#[derive(Debug, Clone)]
pub struct NanoGptConfig {
    /// The vocabulary size.
    pub vocab_size: i64,
    /// The maximum sequence length.
    pub block_size: i64,
    /// The embedding size.
    pub n_embd: i64,
    /// The number of heads.
    pub n_head: i64,
    /// The number of layers.
    pub n_layer: i64,
}

impl NanoGpt {
    /// Create a new NanoGpt
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `config` - The model configuration. See [NanoGptConfig].
    /// # Returns
    /// A new NanoGpt.
    pub fn new(vs: &Path, config: NanoGptConfig) -> Self {
        let block_config = BlockConfig::from(&config);

        let NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_layer,
            ..
        } = config;

        let token_embedding =
            nn::embedding(vs / "embedding", vocab_size, n_embd, Default::default());

        let position_embedding = nn::embedding(
            vs / "position_embedding",
            block_size,
            n_embd,
            Default::default(),
        );

        let mut layers = nn::seq_t();
        for i in 0..n_layer {
            layers = layers.add(Block::new(vs / i, block_config));
        }

        let lm_head = nn::linear(
            vs / "lm_head",
            n_embd,
            vocab_size,
            LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        let ln = layer_norm(
            vs / "ln",
            vec![n_embd],
            LayerNormConfig {
                elementwise_bias: false,
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
            ln,
        }
    }
}

impl ModuleT for NanoGpt {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, t) = xs.size2().unwrap();

        // tok_emb
        let tok_emb = xs.apply(&self.token_embedding); // [batch_size, block_size, n_embd]

        // pos_emb
        let device = xs.device();
        let pos = Tensor::arange_start(0, t, (tch::Kind::Int64, device)); // [t]
        let pos = pos.unsqueeze(0); // [1, t]
        assert_eq!(pos.size(), &[1, t]);

        let pos_emb = pos.apply(&self.position_embedding);
        let pos_emb = pos_emb.view([t, self.n_embd]); // [block_size, n_embd]
        assert_eq!(pos_emb.size(), &[t, self.n_embd]);

        // residual connection
        let x = tok_emb + pos_emb; // [batch_size, block_size, n_embd]
        assert_eq!(x.size(), &[b, t, self.n_embd]);

        // layers
        let x = x.apply_t(&self.layers, train); // [batch_size, block_size, n_embd]

        // layer norm
        let x = x.apply_t(&self.ln, train); // [batch_size, block_size, n_embd]

        // lm head
        let x = x.apply_t(&self.lm_head, train); // [batch_size, block_size, vocab_size]
        assert_eq!(x.size(), &[b, t, self.vocab_size]);
        x
    }
}

impl LMModel for NanoGpt {
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
        let (b, _t) = xs.size2().unwrap();
        // assert_eq!(t, self.block_size);

        // Create a tensor of zeros to store the output
        let outputs = xs.new_zeros(&[b, max_len], (tch::Kind::Int64, xs.device()));

        let mut xs = xs;
        for i in 0..max_len {
            // take the last logits
            let logits = self.forward_t(&xs, false).i((.., -1, ..));
            //  apply softmax to get the probabilities of the next token
            let probs = logits.softmax(-1, tch::Kind::Float);
            // sample the next token
            let next_token = probs.multinomial(1, true);

            // add the next token to the output
            outputs.narrow(1, i, 1).copy_(&next_token);

            // update xs
            xs = Tensor::cat(&[xs, next_token], 1);

            // crop the sequence to the maximum length, starting from the end if needed
            // FUTURE(ssoudan) better way?
            let (_b, t) = xs.size2().unwrap();
            if t > self.block_size {
                xs = xs.i((.., (t - self.block_size)..));
            }
            assert!(xs.size()[1] <= self.block_size);
        }

        outputs
    }
}

/// LMModel is a language model.
pub trait LMModel: ModuleT {
    /// Generate a sequence of tokens from a starting sequence of tokens
    /// and a maximum length.
    ///
    /// xs: the starting sequence of tokens of shape [batch_size, block_size]
    /// max_len: the maximum length of the generated sequence
    /// return: the generated sequence of tokens of shape [batch_size, max_len]
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor;
}

/// Compute the loss
///
/// Use cross entropy loss.
pub fn loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    let (b, t, c) = logits.size3().unwrap();

    let logits = logits.to_kind(tch::Kind::Float);
    let targets = targets.to_kind(tch::Kind::Int64);

    logits
        .view([b * t, c])
        .cross_entropy_for_logits(&targets.view([b * t]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the BigramLanguageModel
    #[test]
    fn test_bigram_language_model() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vocab_size = 100;
        let batch_size = 2;
        let block_size = 5;

        let model = BigramLanguageModel::new(&vs.root(), vocab_size);
        let xs = Tensor::of_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size]);
        println!("xs: {:?}", xs);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size);

        let logits = model.forward(&xs);
        assert_eq!(logits.size(), [batch_size, block_size, vocab_size]);

        let loss = loss(&logits, &xs);
        println!("loss: {:?}", loss);
        // we expect the loss to be close to -ln(1/vocab_size) = 4.17

        // use 0 as start of sequence token - this is '\n' with our data and tokenizer.
        let xs = Tensor::zeros(&[batch_size, 1], (tch::Kind::Int64, tch::Device::Cpu));
        let max_len = 10;
        let ys = model.generate(xs, max_len);
        println!("generated: {:?}", ys);

        // decode the generated sequence of tokens
        let ys = ys.to_kind(tch::Kind::Int64);

        println!("generated: {:?}", ys);

        let first = ys.i((0, ..));
        let first: Vec<i64> = first.into();
        println!("first: {:?}", first);

        let second = ys.i((1, ..));
        let second: Vec<i64> = second.into();
        println!("second: {:?}", second);
    }

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

        let config = NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
        };

        let model = NanoGpt::new(&vs.root(), config);
        let xs = Tensor::of_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size]);
        println!("xs: {:?}", xs);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size);

        let logits = model.forward_t(&xs, false);
        assert_eq!(logits.size(), [batch_size, block_size, vocab_size]);

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

        let config = NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
        };

        let model = NanoGpt::new(&vs.root(), config);
        let xs = Tensor::of_slice(&[0, 1, 2, 3, 4, 5]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size - 2]);
        println!("xs: {:?}", xs);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size - 2);

        let logits = model.forward_t(&xs, false);
        assert_eq!(logits.size(), [batch_size, block_size - 2, vocab_size]);

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

        let config = NanoGptConfig {
            vocab_size,
            block_size,
            n_embd,
            n_head,
            n_layer,
        };

        let model = NanoGpt::new(&vs.root(), config);
        let xs = Tensor::of_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
        let xs = xs.view([batch_size, block_size]);
        println!("xs: {:?}", xs);
        let (b, t) = xs.size2().unwrap();
        assert_eq!(b, batch_size);
        assert_eq!(t, block_size);

        let logits = model.forward_t(&xs, false);
        assert_eq!(logits.size(), [batch_size, block_size, vocab_size]);

        let loss = loss(&logits, &xs);
        println!("loss: {:?}", loss);
        // we expect the loss to be close to -ln(1/vocab_size) = 4.17

        // use 0 as start of sequence token - this is '\n' with our data and tokenizer.
        let xs = Tensor::zeros(&[batch_size, 1], (tch::Kind::Int, tch::Device::Cpu));
        println!("xs: {:?}", xs);
        let max_len = 10;
        let ys = model.generate(xs, max_len);
        println!("generated: {:?}", ys);

        // decode the generated sequence of tokens
        let ys = ys.to_kind(tch::Kind::Int);

        println!("generated: {:?}", ys);

        let first = ys.i((0, ..));
        let first: Vec<i64> = first.into();
        println!("first: {:?}", first);

        let second = ys.i((1, ..));
        let second: Vec<i64> = second.into();
        println!("second: {:?}", second);
    }
}
