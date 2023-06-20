use std::fmt::Display;

use clap::{Parser, Subcommand, ValueEnum};
/// Torch device to use.
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
pub enum Device {
    /// CPU
    #[default]
    Cpu,
    /// CUDA if available
    Cuda,
    /// MPS
    #[cfg(target_arch = "aarch64")]
    Mps,
}

/// Arguments for the NanoGPT model.
#[derive(Parser, Debug, Clone)]
pub struct NanoGptArgs {
    /// Maximum sequence length
    #[arg(long, default_value_t = 256)]
    pub block_size: usize,
    /// the size of the embedding to use
    #[arg(short = 'E', long, default_value_t = 384)]
    pub n_embd: i64,
    /// Number of layers
    #[arg(short = 'L', long, default_value_t = 6)]
    pub n_layer: i64,
    /// Number of heads
    #[arg(short = 'H', long, default_value_t = 6)]
    pub n_head: i64,
    /// Dropout probability
    #[arg(short = 'D', long, default_value_t = 0.2)]
    pub dropout: f64,
    /// Biases - default is true
    #[arg(short = 'B', long, default_value_t = false)]
    pub bias: bool,

    /// Tie the weights of the token embedding and the lm head
    #[arg(long, default_value_t = false)]
    pub tie_weights: bool,

    /// Vocabulary file
    #[arg(long, default_value = "data/input.txt")]
    pub vocab_file: String,
}

impl Default for NanoGptArgs {
    fn default() -> Self {
        Self {
            block_size: 256,
            n_embd: 384,
            n_layer: 6,
            n_head: 6,
            dropout: 0.2,
            bias: true,
            tie_weights: true,
            vocab_file: "data/input.txt".to_string(),
        }
    }
}

/// Training parameters.
#[derive(Parser, Debug, Clone)]
pub struct TrainingParameters {
    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    /// Number of epochs
    #[arg(long, default_value_t = 1.)]
    pub n_epochs: f32,
    /// Batch size
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// Number of training iterations before evaluating the losses on the
    /// training and validation sets.
    #[arg(long, default_value_t = 1000)]
    pub steps_between_loss_estimation: usize,
    /// Number of batches to use for estimating the losses.
    #[arg(long, default_value_t = 100)]
    pub loss_estimation_steps: usize,
    /// Maximum gradient norm - 0.0 means no clipping.
    #[arg(long, default_value_t = 1.0)]
    pub max_grad_norm: f64,

    /// Final checkpoint path
    #[arg(long)]
    pub final_checkpoint_path: Option<String>,

    /// Dataset path
    #[arg(long, default_value = "data/input.txt")]
    pub dataset_path: String,

    /// Validation dataset path
    #[arg(long)]
    pub validation_dataset_path: Option<String>,

    /// Prompt to use for an example after the training
    #[arg(long)]
    pub prompt: Option<String>,

    /// Rng seed for the dataloader
    #[arg(long, default_value_t = 142)]
    pub dataloader_rng_seed: u64,

    /// Name of the run
    #[arg(long)]
    pub run_name: Option<String>,

    /// Path to the tensorboard directory
    /// If not provided, no tensorboard logging will be done.
    #[arg(long)]
    pub tensorboard_dir: Option<String>,
}

/// GPT2 model size.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Gpt2Size {
    /// Normal
    #[clap(name = "gpt2")]
    Normal,
    /// Medium
    #[clap(name = "gpt2-medium")]
    Medium,
    /// Large
    #[clap(name = "gpt2-large")]
    Large,
    /// XLarge
    #[clap(name = "gpt2-xl")]
    XLarge,
}

impl Display for Gpt2Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gpt2Size::Normal => write!(f, "gpt2"),
            Gpt2Size::Medium => write!(f, "gpt2-medium"),
            Gpt2Size::Large => write!(f, "gpt2-large"),
            Gpt2Size::XLarge => write!(f, "gpt2-xl"),
        }
    }
}

/// Arguments for the GPT2 model.
#[derive(Parser, Debug, Clone)]
pub struct GPT2Args {
    /// The size of the model to use.
    #[arg(short = 'S', long, default_value_t = Gpt2Size::Normal)]
    pub size: Gpt2Size,

    /// Tokenizer JSON file
    #[arg(long, default_value = "models/gpt2/tokenizer.json")]
    pub tokenizer_path: String,
}

/// The model to use.
#[derive(Subcommand, Debug, Clone)]
pub enum Model {
    /// NanoGPT
    NanoGpt {
        /// The arguments for the NanoGPT model
        #[command(flatten)]
        args: NanoGptArgs,
    },
    /// Bigram
    Bigram {
        /// Vocabulary file
        #[arg(long, default_value = "data/input.txt")]
        vocab_file: String,
    },
    /// GPT2
    GPT2 {
        /// The arguments for the GPT2 model
        #[command(flatten)]
        args: GPT2Args,
    },
}

impl Default for Model {
    fn default() -> Self {
        Self::NanoGpt {
            args: NanoGptArgs::default(),
        }
    }
}

/// The configuration for the learning process
pub struct LearnConfig {
    /// The number of epochs to train
    pub n_epochs: f32,
    /// The learning rate
    pub lr: f64,
    /// The number of steps between loss estimation
    pub steps_between_loss_estimation: usize,
    /// The number of steps to estimate the loss
    pub loss_estimation_steps: usize,
    /// The maximum gradient norm
    pub max_grad_norm: Option<f64>,
}
