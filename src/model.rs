use std::borrow::Borrow;

use tch::nn::{init, Init, ModuleT};
use tch::{nn, IndexOp, Kind, Tensor};

/// BigramLanguageModel is a language model that uses the current token to
/// predict the next token.
#[derive(Debug)]
pub struct BigramLanguageModel {
    /// The embedding layer
    embedding: nn::Embedding,
}

impl BigramLanguageModel {
    /// Create a new BigramLanguageModel
    pub fn new(vs: &nn::Path, vocab_size: i64) -> Self {
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

    fn block_size(&self) -> usize {
        256 // FUTURE(ssoudan) ??
    }
}

impl nn::ModuleT for BigramLanguageModel {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// Head configuration
#[derive(Debug, Clone, Copy)]
pub struct HeadConfig {
    /// The size of the head.
    pub head_size: i64,
    /// Block size.
    pub block_size: usize,
    /// The embedding size.
    pub n_embd: i64,
    /// Whether to use bias.
    pub bias: bool,
    /// Dropout probability.
    pub dropout: f64,
}

impl From<&MultiHeadSelfAttentionConfig> for HeadConfig {
    fn from(config: &MultiHeadSelfAttentionConfig) -> Self {
        Self {
            head_size: config.head_size,
            block_size: config.block_size,
            n_embd: config.n_embd,
            bias: config.bias,
            dropout: config.dropout,
        }
    }
}

/// One head self-attention.
#[derive(Debug)]
struct Head {
    // key: nn::Linear,
    // query: nn::Linear,
    // value: nn::Linear,
    c_attn: nn::Linear,
    mask: Tensor,
    dropout: f64,
    head_size: i64,
}

#[allow(unused)]
impl Head {
    /// Create a new Head.
    ///
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `config` - The configuration. See [`HeadConfig`].
    ///
    /// # Returns
    /// A new Head.
    ///
    /// # Notes
    /// The input of `forward` is expected to be of shape `[batch_size,
    /// block_size, C]`.
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(vs: T, config: HeadConfig) -> Self {
        let HeadConfig {
            head_size,
            block_size,
            n_embd,
            bias,
            dropout,
        } = config;

        let vs = vs.borrow();

        let device = vs.device();

        // single qkv linear layer and split
        let c_attn = nn::linear(
            vs / "c_attn",
            n_embd,
            head_size * 3,
            nn::LinearConfig {
                bias,
                ..Default::default()
            },
        );

        let mask = Tensor::ones(
            [(block_size as i64), (block_size as i64)],
            (tch::Kind::Float, device),
        )
        .tril(0);
        Self {
            // key,
            // query,
            // value,
            c_attn,
            mask,
            head_size,
            dropout,
        }
    }
}

impl nn::ModuleT for Head {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, t, _c) = xs.size3().unwrap();
        let head_size = self.head_size;

        // the dimension of the keys
        let d_k = head_size;

        // use a single linear layer and split (in q, k, v order)
        let c_attn = xs.apply(&self.c_attn); // [b, t, head_size * 3]
        let qkv = c_attn.split(self.head_size, 2); // 3 * [b, t, head_size]
        let q = &qkv[0];
        let k = &qkv[1];
        let v = &qkv[2];

        // let k = xs.apply(&k); // [b, t, head_size]
        assert_eq!(k.size(), &[b, t, head_size]);
        // let q = xs.apply(&self.query); // [b, t, head_size]
        assert_eq!(q.size(), &[b, t, head_size]);
        // let v = xs.apply(&self.value); // [b, t, head_size]
        assert_eq!(v.size(), &[b, t, head_size]);

        // Attention scores
        let wei = q.matmul(&k.transpose(-2, -1)) / (d_k as f64).sqrt();
        let wei = wei.masked_fill(&self.mask.i((..t, ..t)).eq(0.), f64::NEG_INFINITY);
        assert_eq!(wei.size(), &[b, t, t]);

        let wei = wei.softmax(-1, tch::Kind::Float);
        assert_eq!(wei.size(), &[b, t, t]);

        let wei = wei.dropout(self.dropout, train);

        // weighted aggregation of the values
        let out = wei.matmul(&v); // [b, t, head_size]
        assert_eq!(out.size(), &[b, t, head_size]);

        out
    }
}

/// Configuration for the MultiHeadSelfAttention layer.
#[derive(Debug, Clone, Copy)]
pub struct MultiHeadSelfAttentionConfig {
    /// The maximum sequence length.
    pub block_size: usize,
    /// The embedding size.
    pub n_embd: i64,
    /// The size of the head.
    pub head_size: i64,
    /// The number of heads.
    pub n_head: i64,
    /// Bias flag.
    pub bias: bool,
    /// Dropout probability.
    pub dropout: f64,
}

impl From<&BlockConfig> for MultiHeadSelfAttentionConfig {
    fn from(config: &BlockConfig) -> Self {
        Self {
            block_size: config.block_size,
            n_embd: config.n_embd,
            head_size: config.head_size,
            n_head: config.n_head,
            bias: config.bias,
            dropout: config.dropout,
        }
    }
}

/// MultiHeadSelfAttention is a multi-head self-attention layer.
#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    c_attn: nn::Linear,
    projection: nn::Linear,
    dropout: f64,
    n_embd: i64,
    n_head: i64,
    head_size: i64,
    mask: Tensor,
}

impl MultiHeadSelfAttention {
    /// Create a new MultiHeadSelfAttention.
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `config` - The configuration. See [`MultiHeadSelfAttentionConfig`].
    /// # Returns
    /// A new MultiHeadSelfAttention.
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(vs: T, config: MultiHeadSelfAttentionConfig) -> Self {
        let MultiHeadSelfAttentionConfig {
            n_embd,
            n_head,
            bias,
            dropout,
            ..
        } = config;

        let head_config = HeadConfig::from(&config);
        let head_size = head_config.head_size;

        let vs = vs.borrow();

        // Build attention heads all at once
        ////////////////////////////////////////
        let attn_vs = vs / "c_attn";

        // create manually, init and transpose so it matches the
        // checkpoints from gpt2
        let c_attn_ws = attn_vs.var(
            "weight",
            &[n_embd, n_embd * 3],
            init::DEFAULT_KAIMING_UNIFORM,
        );

        let c_attn_bs = if bias {
            Some(attn_vs.var("bias", &[n_embd * 3], Init::Const(0.)))
        } else {
            None
        };

        // transpose - to align with GPT2 checkpoints
        let c_attn_ws = c_attn_ws.transpose(0, 1);

        assert_eq!(c_attn_ws.size(), &[3 * n_embd, n_embd]);
        if bias {
            assert_eq!(c_attn_bs.as_ref().unwrap().size(), &[3 * n_embd]);
        } else {
            assert!(c_attn_bs.is_none());
        }

        let c_attn = nn::Linear {
            ws: c_attn_ws,
            bs: c_attn_bs,
        };

        // Build projections
        ////////////////////////////////////////
        let projection_vs = vs / "c_proj";

        let c_proj_w =
            projection_vs.var("weight", &[n_embd, n_embd], init::DEFAULT_KAIMING_UNIFORM);

        let c_proj_b = if bias {
            Some(projection_vs.var("bias", &[n_embd], Init::Const(0.)))
        } else {
            None
        };

        // transpose - to align with GPT2 checkpoints
        let c_proj_w = c_proj_w.transpose(0, 1);

        assert_eq!(c_proj_w.size(), &[n_embd, n_embd]);
        if bias {
            assert_eq!(c_proj_b.as_ref().unwrap().size(), &[n_embd]);
        } else {
            assert!(c_proj_b.is_none());
        }

        let projection = nn::Linear {
            ws: c_proj_w,
            bs: c_proj_b,
        };

        let block_size = config.block_size as i64;
        let device = vs.device();

        let mask = Tensor::ones([block_size, block_size], (Kind::Float, device))
            .tril(0)
            .view([1, 1, block_size, block_size]);

        Self {
            c_attn,
            projection,
            dropout,
            mask,
            n_embd,
            n_head,
            head_size,
        }
    }
}

impl nn::ModuleT for MultiHeadSelfAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, t, c) = xs.size3().unwrap();

        // got to process all the 'heads' at once - and each head does the query, key,
        // value at once too
        let qkv = self.c_attn.forward_t(xs, train).split(self.n_embd, 2);
        let q = qkv[0].view([b, t, self.n_head, -1]).transpose(1, 2); // [b, n_head, t, head_size]
        let k = qkv[1].view([b, t, self.n_head, -1]).transpose(1, 2); // [b, n_head, t, head_size]
        let v = qkv[2].view([b, t, self.n_head, -1]).transpose(1, 2); // [b, n_head, t, head_size]

        assert_eq!(q.size(), &[b, self.n_head, t, self.head_size]);
        assert_eq!(k.size(), &[b, self.n_head, t, self.head_size]);
        assert_eq!(v.size(), &[b, self.n_head, t, self.head_size]);

        // compute attention score
        let wei = q.matmul(&k.transpose(-2, -1)) / (self.head_size as f64).sqrt(); // [b, n_head, t, t]
        assert_eq!(wei.size(), &[b, self.n_head, t, t]);
        let wei = wei.masked_fill(&self.mask.i((.., .., ..t, ..t)).eq(0.), f64::NEG_INFINITY);
        let wei = wei.softmax(-1, Kind::Float); // FUTURE(ssoudan) Float?
        assert_eq!(wei.size(), &[b, self.n_head, t, t]);
        let wei = wei.dropout(self.dropout, train);
        let attn = wei.matmul(&v); // [b, n_head, t, head_size]

        assert_eq!(attn.size(), &[b, self.n_head, t, self.head_size]);

        let heads = attn.transpose(1, 2).contiguous().view([b, t, c]);

        heads.apply(&self.projection).dropout(self.dropout, train)
    }
}

/// Feed forward layer configuration.
#[derive(Debug, Clone, Copy)]
pub struct FeedForwardConfig {
    /// The embedding size.
    pub n_embd: i64,
    /// The dropout probability.
    pub dropout: f64,
    /// Whether to add a bias after the linear transformation.
    pub bias: bool,
}

impl From<&BlockConfig> for FeedForwardConfig {
    fn from(config: &BlockConfig) -> Self {
        Self {
            n_embd: config.n_embd,
            dropout: config.dropout,
            bias: config.bias,
        }
    }
}

/// Feed forward layer
#[derive(Debug)]
pub struct FeedForward {
    net: nn::SequentialT,
}

impl FeedForward {
    /// Create a new FeedForward
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(vs: T, config: FeedForwardConfig) -> Self {
        let FeedForwardConfig {
            n_embd,
            dropout,
            bias,
        } = config;

        // c_fc linear
        let c_fc_vs = vs.borrow() / "c_fc";
        let c_fc_weight = c_fc_vs.var(
            "weight",
            &[n_embd, 4 * n_embd],
            init::DEFAULT_KAIMING_UNIFORM,
        );

        let c_fc_bias = if bias {
            Some(c_fc_vs.var("bias", &[4 * n_embd], Init::Const(0.)))
        } else {
            None
        };

        // transpose - to align with GPT2 checkpoints
        let c_fc_weight = c_fc_weight.transpose(0, 1);

        assert_eq!(c_fc_weight.size(), &[4 * n_embd, n_embd]);

        if bias {
            assert_eq!(c_fc_bias.as_ref().unwrap().size(), &[4 * n_embd]);
        } else {
            assert!(c_fc_bias.is_none());
        }

        let c_fc = nn::Linear {
            ws: c_fc_weight,
            bs: c_fc_bias,
        };

        // c_proj linear
        let c_proj_vs = vs.borrow() / "c_proj";
        let c_proj_weight = c_proj_vs.var(
            "weight",
            &[4 * n_embd, n_embd],
            init::DEFAULT_KAIMING_UNIFORM,
        );

        let c_proj_bias = if bias {
            Some(c_proj_vs.var("bias", &[n_embd], Init::Const(0.)))
        } else {
            None
        };

        // transpose - to align with GPT2 checkpoints
        let c_proj_weight = c_proj_weight.transpose(0, 1);

        assert_eq!(c_proj_weight.size(), &[n_embd, 4 * n_embd]);

        if bias {
            assert_eq!(c_proj_bias.as_ref().unwrap().size(), &[n_embd]);
        } else {
            assert!(c_proj_bias.is_none());
        }

        let c_proj = nn::Linear {
            ws: c_proj_weight,
            bs: c_proj_bias,
        };

        let net = nn::seq_t()
            .add(c_fc)
            .add_fn(|xs| xs.gelu("tanh"))
            .add(c_proj)
            .add_fn_t(move |xs, train| xs.dropout(dropout, train));

        Self { net }
    }
}

impl nn::ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.net.forward_t(xs, train)
    }
}

/// Block configuration.
#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    /// The maximum sequence length.
    pub block_size: usize,
    /// The embedding size.
    pub n_embd: i64,
    /// The size of the head.
    pub head_size: i64,
    /// The number of heads.
    pub n_head: i64,
    /// Bias flag.
    pub bias: bool,
    /// Dropout probability.
    pub dropout: f64,
}

impl From<&NanoGptConfig> for BlockConfig {
    fn from(config: &NanoGptConfig) -> Self {
        Self {
            block_size: config.block_size,
            n_embd: config.n_embd,
            head_size: config.n_embd / config.n_head,
            n_head: config.n_head,
            bias: config.bias,
            dropout: config.dropout,
        }
    }
}

/// LayerNorm configuration.
#[derive(Debug, Clone, Copy)]
pub struct LayerNormConfig {
    /// Whether to use CUDNN.
    pub cudnn_enabled: bool,
    /// A small constant added to the denominator for numerical stability.
    pub eps: f64,
    /// Whether to apply a linear transformation.
    pub elementwise_linear: bool,
    /// Whether to apply a bias.
    pub elementwise_bias: bool,
    /// The weight initialization.
    pub ws_init: nn::Init,
    /// The bias initialization.
    pub bs_init: nn::Init,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            elementwise_linear: true,
            elementwise_bias: true,
            eps: 1e-5,
            cudnn_enabled: true,
            ws_init: nn::Init::Const(1.),
            bs_init: nn::Init::Const(0.),
        }
    }
}

/// Layer normalization with optionally no biases.
///
/// - See [Layer Normalization](https://arxiv.org/abs/1607.06450).
/// - See [`nn::LayerNorm`].
#[derive(Debug)]
pub struct LayerNorm {
    /// The configuration.
    config: LayerNormConfig,
    /// The weight.
    ws: Option<Tensor>,
    /// The bias.
    bs: Option<Tensor>,
    /// The normalized shape.
    normalized_shape: Vec<i64>,
}

impl LayerNorm {
    /// Create a new LayerNorm.
    ///
    /// # Arguments
    /// * `vs` - The variable store.
    /// * `normalized_shape` - The shape of the normalized tensor.
    /// * `config` - The configuration.
    /// # Returns
    /// The LayerNorm.
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(
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

        Self {
            config,
            ws,
            bs,
            normalized_shape,
        }
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

/// A Block is a transformer decoder block.
#[derive(Debug)]
struct Block {
    /// SA heads
    attn: MultiHeadSelfAttention,
    /// Feed forward layer
    mlp: FeedForward,
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
    pub fn new<'a, T: Borrow<nn::Path<'a>>>(vs: T, config: BlockConfig) -> Self {
        let vs = vs.borrow();

        let mhsa_config = MultiHeadSelfAttentionConfig::from(&config);
        let ffwd_config = FeedForwardConfig::from(&config);

        let BlockConfig { n_embd, n_head, .. } = config;

        assert_eq!(n_embd % n_head, 0, "n_emb must be divisible by n_head");

        let sa_heads = MultiHeadSelfAttention::new(vs / "attn", mhsa_config);
        let mlp = FeedForward::new(vs / "mlp", ffwd_config);

        let ln1 = LayerNorm::new(
            vs / "ln_1",
            vec![n_embd],
            LayerNormConfig {
                elementwise_bias: config.bias,
                ..Default::default()
            },
        );
        let ln2 = LayerNorm::new(
            vs / "ln_2",
            vec![n_embd],
            LayerNormConfig {
                elementwise_bias: config.bias,
                ..Default::default()
            },
        );

        Self {
            attn: sa_heads,
            mlp,
            ln1,
            ln2,
        }
    }
}

impl nn::ModuleT for Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // SA heads with residual connection
        let xs = xs + xs.apply_t(&self.ln1, train).apply_t(&self.attn, train);
        // [b, t, n_embd]

        // Feed forward layer with residual connection
        &xs + &xs.apply_t(&self.ln2, train).apply_t(&self.mlp, train)
        // [b, t, n_embd]
    }
}

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
    ln: LayerNorm,
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

        let ln = LayerNorm::new(
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
            ln,
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
        let outputs = xs.new_zeros([b, max_len], (tch::Kind::Int64, xs.device()));

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
            // NOFUTURE(ssoudan) better way?
            let (_b, t) = xs.size2().unwrap();
            if t > self.block_size as i64 {
                xs = xs.i((.., (t - (self.block_size as i64))..));
            }
            assert!(xs.size()[1] <= self.block_size as i64);
        }

        outputs
    }

    fn block_size(&self) -> usize {
        self.block_size
    }
}

/// LMModel is a language model.
pub trait LMModel: nn::ModuleT {
    /// Generate a sequence of tokens from a starting sequence of tokens
    /// and a maximum length.
    ///
    /// xs: the starting sequence of tokens of shape [batch_size, block_size]
    /// max_len: the maximum length of the generated sequence
    /// return: the generated sequence of tokens of shape [batch_size, max_len]
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor;

    /// Return the block size
    fn block_size(&self) -> usize;
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
    use tch::nn::{Module, ModuleT};

    use super::*;

    /// Test the BigramLanguageModel
    #[test]
    fn test_bigram_language_model() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vocab_size = 100;
        let batch_size = 2;
        let block_size = 5;

        let model = BigramLanguageModel::new(&vs.root(), vocab_size);
        let xs = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int);
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
        let xs = Tensor::zeros([batch_size, 1], (tch::Kind::Int64, tch::Device::Cpu));
        let max_len = 10;
        let ys = model.generate(xs, max_len);
        println!("generated: {:?}", ys);

        // decode the generated sequence of tokens
        let ys = ys.to_kind(tch::Kind::Int64);

        println!("generated: {:?}", ys);

        let first = ys.i((0, ..));
        let first: Vec<i64> = first.reshape(-1).try_into().unwrap();
        println!("first: {:?}", first);

        let second = ys.i((1, ..));
        let second: Vec<i64> = second.reshape(-1).try_into().unwrap();
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
        println!("xs: {:?}", xs);
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
        println!("xs: {:?}", xs);
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

    /// Test shared tensor for linear layer
    #[test]
    fn test_shared_tensor_linear() {
        let vs = nn::VarStore::new(tch::Device::Cpu);

        let input_size = 10;
        let output_size = 20;
        let n = 2;
        let batch_size = 2;

        let shared_linear_weight = Tensor::randn(
            &[input_size, n * output_size],
            (tch::Kind::Float, tch::Device::Cpu),
        );

        // transpose
        let shared_linear_weight = shared_linear_weight.transpose(0, 1);

        let linear_weights = shared_linear_weight.split(output_size, 0);

        let linears = linear_weights
            .into_iter()
            .map(|ws| nn::Linear { ws, bs: None })
            .collect::<Vec<_>>();

        // Test forward pass
        let xs = Tensor::randn(
            &[batch_size, input_size],
            (tch::Kind::Float, tch::Device::Cpu),
        );

        let mut ys = Vec::new();

        for linear in &linears {
            let y = linear.forward(&xs);
            assert_eq!(y.size(), [batch_size, output_size]);
            ys.push(y);
        }

        assert_eq!(ys.len(), n as usize);

        assert_eq!(ys[0].size(), [batch_size, output_size]);
        assert_eq!(ys[1].size(), [batch_size, output_size]);
    }
}
