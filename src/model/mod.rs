/// Bigram model.
pub mod bigram;
/// Nanogpt model.
pub mod nano;

use std::borrow::Borrow;

use nano::NanoGptConfig;
use tch::nn::{init, Init};
use tch::{nn, IndexOp, Kind, Tensor};

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
        let wei = q.matmul(&k.transpose(-2, -1)) * (1. / (self.head_size as f64).sqrt()); // [b, n_head, t, t]
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
        let (b, t, n_embd) = xs.size3().unwrap();
        // SA heads with residual connection
        let xs = xs + xs.apply_t(&self.ln1, train).apply_t(&self.attn, train);
        assert_eq!(xs.size(), &[b, t, n_embd]);
        // [b, t, n_embd]

        // Feed forward layer with residual connection
        let xs = &xs + &xs.apply_t(&self.ln2, train).apply_t(&self.mlp, train);
        assert_eq!(xs.size(), &[b, t, n_embd]);
        xs
        // [b, t, n_embd]
    }
}

// FUTURE(ssoudan) support JIT for models

/// LanguageModel is a language model.
pub trait LanguageModel: nn::ModuleT {
    /// Generate a sequence of tokens from a starting sequence of tokens
    /// and a maximum length.
    ///
    /// # Arguments
    /// - xs: the starting sequence of tokens of shape \[batch_size, x\]; x <=
    /// `block_size`
    /// - max_len: the maximum length of the generated sequence
    ///
    /// # Returns
    /// the generated sequence of tokens of shape \[batch_size, max_len\]
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor;

    /// Return the block size which is the maximum length of the input.
    fn block_size(&self) -> usize;

    /// Compute the probability of the next token
    ///
    /// # Arguments
    /// - xs: the sequence of input tokens of shape \[batch_size, x\]; x <=
    /// `block_size`
    ///
    /// # Returns
    /// the probability of the next token of shape \[batch_size, vocab_size\]
    /// as a detached tensor.
    fn probabilities(&self, xs: &Tensor) -> Tensor;
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

/// Utils to load GPT2 safetensors files.
pub mod utils {
    use tch::nn;
    use tch::nn::VarStore;

    /// Load tensors from a file and transpose them if needed.
    pub fn load_safetensors(
        mut vs: VarStore,
        var_to_transpose: &[&str; 4],
        source: &str,
        prefix: &str,
    ) -> VarStore {
        let mut load_vs = nn::VarStore::new(tch::Device::Cpu);
        // recreate the variables with the same name in vs2
        prepare_to_load(&mut load_vs, &mut vs, var_to_transpose, prefix);
        let _ = load_vs.load_partial(source);

        // print the values of the variables in vs
        for (name, variable) in load_vs.variables() {
            println!("{}: {:?}", name, variable);
        }

        // copy the values from vs2 to vs
        copy_from(&mut load_vs, &mut vs, var_to_transpose, prefix);
        vs
    }

    fn copy_from(
        source: &mut VarStore,
        dest: &mut VarStore,
        var_to_transpose: &[&str; 4],
        prefix: &str,
    ) {
        for (name, mut variable) in dest.variables() {
            let name_slit = name.split('.');
            let var_name = name_slit.clone().last().unwrap();

            let vs2_root = name_slit
                .clone()
                .take(name_slit.clone().count() - 1)
                .fold(source.root(), |vs, name| vs / name);

            let vs2_var = vs2_root.get(var_name).unwrap();

            // remove the prefix
            let name = name.strip_prefix(prefix).unwrap_or_else(|| name.as_ref());
            // transpose if needed
            if var_to_transpose.contains(&name) {
                let vs2_var = vs2_var.transpose(0, 1);
                tch::no_grad(|| variable.copy_(&vs2_var));
            } else {
                tch::no_grad(|| variable.copy_(&vs2_var));
            }
        }
    }

    /// populate a varstore with the same variables as vs (transposed if
    /// needed)
    fn prepare_to_load(
        new_store: &mut VarStore,
        vs: &mut VarStore,
        var_to_transpose: &[&str],
        prefix: &str,
    ) {
        for (name, variable) in vs.variables() {
            // skip variables that do not start with prefix
            if !name.starts_with(prefix) {
                continue;
            }

            // remove prefix for the comparison in the next step
            let name_ = name.strip_prefix(prefix).unwrap_or_else(|| name.as_ref());

            let dims = if var_to_transpose.contains(&name_) {
                let dims = variable.size();
                vec![dims[1], dims[0]]
            } else {
                variable.size()
            };

            let p = name.split('.');
            let split_count = p.clone().count();
            let var_name = p.last().unwrap();
            let p = name.split('.');
            // remove last element
            let p = p.take(split_count - 1);
            let vs_var = p.fold(new_store.root(), |vs, name| vs / name);

            // create the variable with the same name in vs2 with the same size - transpose
            // if needed
            let _ = vs_var.ones_no_train(var_name, &dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::nn::{Module, ModuleT};
    use tch::{nn, Kind, Tensor};

    use crate::model::utils::load_safetensors;
    use crate::model::{Block, BlockConfig};

    /// Test shared tensor for linear layer
    #[test]
    fn test_shared_tensor_linear() {
        let device = tch::Device::Cpu;
        let input_size = 10;
        let output_size = 20;
        let n = 2;
        let batch_size = 2;

        let shared_linear_weight =
            Tensor::randn([input_size, n * output_size], (tch::Kind::Float, device));

        // transpose
        let shared_linear_weight = shared_linear_weight.transpose(0, 1);
        assert_eq!(shared_linear_weight.size(), [n * output_size, input_size]);

        let linear_weights = shared_linear_weight.split(output_size, 0);

        let linears = linear_weights
            .into_iter()
            .map(|ws| nn::Linear { ws, bs: None })
            .collect::<Vec<_>>();

        // Test forward pass
        let xs = Tensor::randn([batch_size, input_size], (tch::Kind::Float, device));

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

    #[test]
    fn test_block() {
        let device = tch::Device::Cpu;
        let vs = nn::VarStore::new(device);

        let config = BlockConfig {
            block_size: 1024,
            n_embd: 768,
            head_size: 768 / 12,
            n_head: 12,
            bias: true,
            dropout: 0.0,
        };

        let prefix = "h.0";
        let prefix_ = prefix.clone().split('.');
        let root = prefix_.fold(vs.root(), |vs, name| vs / name);

        let block = Block::new(root.clone(), config);

        let input_size = 128;

        let var_to_transpose = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ];

        // print h.0.ln_1.weight in the varstore
        // let var_name = "ln_1.weight";
        // let vs_var = vs
        //     .root()
        //     .sub("h")
        //     .sub("0")
        //     .sub("ln_1")
        //     .get("weight")
        //     .unwrap();
        // println!(
        //     "before loading: {} in {}",
        //     var_name,
        //     vs_var.i(..10).to_string(2).unwrap()
        // );

        let source = "models/gpt2/model.safetensors";
        let vs = load_safetensors(vs, &var_to_transpose, source, prefix);

        // print h.0.ln_1.weight in the varstore
        // let var_name = "ln_1.weight";
        // let vs_var = vs
        //     .root()
        //     .sub("h")
        //     .sub("0")
        //     .sub("ln_1")
        //     .get("weight")
        //     .unwrap();
        // println!(
        //     "after loading: {} in {}",
        //     var_name,
        //     vs_var.i(..10).to_string(2).unwrap()
        // );

        // now we can use the block
        let x = vs.root().ones_no_train("x", &[1, input_size, 768]);

        let x_sum = x.sum(Kind::Float);
        let x_sum: f64 = x_sum.try_into().unwrap();
        println!("x_sum: {}", x_sum);
        assert!((x_sum - 98304.0).abs() < 1e-6);

        let y_2 = block.forward_t(&x, false);
        assert_eq!(y_2.size(), &[1, input_size, 768]);

        let y_2_sum = y_2.sum(Kind::Float);
        let y_2_sum: f64 = y_2_sum.try_into().unwrap();
        println!("y_2_sum: {}", y_2_sum);
        assert!((y_2_sum - 95120.4296875).abs() < 1e-6);
    }
}
