//! NanoGPTRS: A rust implementation of the NanoGPT
use std::fmt::Display;

use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use nanogptrs::data::{load_file, Gpt2Tokenizer, Loader, TokenizedData, Tokenizer, Vocab};
use nanogptrs::estimate::LossEstimator;
use nanogptrs::learn::logger::TensorboardReporter;
use nanogptrs::learn::{Observer, PbProgressReporter, ProgressReporter};
use nanogptrs::model::{loss, LMModel, NanoGptConfig};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use tch::nn::{OptimizerConfig, VarStore};
use tch::Tensor;

// FUTURE(ssoudan): weight decay
// FUTURE(ssoudan): lr scheduler

/// Torch device to use.
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
enum Device {
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
struct NanoGptArgs {
    /// Maximum sequence length
    #[arg(long, default_value_t = 256)]
    block_size: usize,
    /// the size of the embedding to use
    #[arg(short = 'E', long, default_value_t = 384)]
    n_embd: i64,
    /// Number of layers
    #[arg(short = 'L', long, default_value_t = 6)]
    n_layer: i64,
    /// Number of heads
    #[arg(short = 'H', long, default_value_t = 6)]
    n_head: i64,
    /// Dropout probability
    #[arg(short = 'D', long, default_value_t = 0.2)]
    dropout: f64,
    /// Biases - default is true
    #[arg(short = 'B', long, default_value_t = true)]
    bias: bool,

    /// Tie the weights of the token embedding and the lm head
    #[arg(long, default_value_t = true)]
    tie_weights: bool,

    /// Vocabulary file
    #[arg(long, default_value = "data/input.txt")]
    vocab_file: String,
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
struct TrainingParameters {
    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
    /// Number of epochs
    #[arg(long, default_value_t = 1.)]
    n_epochs: f32,
    /// Batch size
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Number of training iterations before evaluating the losses on the
    /// training and validation sets.
    #[arg(long, default_value_t = 1000)]
    steps_between_loss_estimation: usize,
    /// Number of batches to use for estimating the losses.
    #[arg(long, default_value_t = 100)]
    loss_estimation_steps: usize,
    /// Maximum gradient norm - 0.0 means no clipping.
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f64,

    /// Final checkpoint path
    #[arg(long)]
    final_checkpoint_path: Option<String>,

    /// Dataset path
    #[arg(long, default_value = "data/input.txt")]
    dataset_path: String,

    /// Prompt to use for an example after the training
    #[arg(long)]
    prompt: Option<String>,

    /// Rng seed for the dataloader
    #[arg(long, default_value_t = 142)]
    dataloader_rng_seed: u64,

    /// Name of the run
    #[arg(long)]
    run_name: Option<String>,

    /// Path to the tensorboard directory
    /// If not provided, no tensorboard logging will be done.
    #[arg(long)]
    tensorboard_dir: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Gpt2Size {
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
struct GPT2Args {
    /// The size of the model to use.
    #[arg(short = 'S', long, default_value_t = Gpt2Size::Normal)]
    size: Gpt2Size,

    /// Tokenizer JSON file
    #[arg(long, default_value = "models/gpt2/tokenizer.json")]
    tokenizer_path: String,
}

/// The model to use.
#[derive(Subcommand, Debug, Clone)]
enum Model {
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

/// The action to perform.
#[derive(Subcommand, Debug, Clone)]
enum Action {
    /// Next tokens distribution
    NextToken {
        /// The model to use
        #[command(subcommand)]
        model: Model,

        /// The prompt to use
        #[arg(long, default_value = "Once upon a time ")]
        prompt: String,
    },
    /// Generate text
    Generate {
        /// The model to use
        #[command(subcommand)]
        model: Model,

        /// The number of tokens to generate
        #[arg(long, default_value_t = 128)]
        max_len: usize,

        /// The prompt to use
        #[arg(long, default_value = "Once upon a time ")]
        prompt: String,
    },
    /// Train the model
    Train {
        /// The model to use
        #[command(subcommand)]
        model: Model,

        /// Training parameters
        #[clap(flatten)]
        training_params: TrainingParameters,
    },
}

impl Default for Action {
    fn default() -> Self {
        Self::Generate {
            model: Model::default(),
            max_len: 128,
            prompt: "Once upon a time ".to_string(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The device to use
    #[arg(short, long, default_value = "cuda")]
    device: Device,

    /// Checkpoint to restore from
    #[arg(short, long)]
    restore_from: Option<String>,
    /// The action to perform
    #[command(subcommand)]
    action: Action,
}

fn main() {
    let args = Args::parse();

    let device = match args.device {
        Device::Cpu => tch::Device::Cpu,
        Device::Cuda => tch::Device::cuda_if_available(),
        #[cfg(target_arch = "aarch64")]
        Device::Mps => tch::Device::Mps,
    };

    // print a banner with nanoGPTrs
    println!(r#"nanoGPTrs"#);

    // if not built in release mode, print a big warning
    #[cfg(debug_assertions)]
    {
        println!("WARNING: This is a debug build. Things will be painfully slow.");
    }

    ///////

    let mut vs = tch::nn::VarStore::new(device);

    match args.action {
        Action::NextToken { model, prompt } => {
            // Build the model
            let (model, tokenizer) = create_model(&mut vs, model);
            println!("[+] Got a model and a tokenizer");

            // Restore the model from the checkpoint
            if args.restore_from.is_some() {
                println!("[.] Restoring the model from the checkpoint");
                // Restore the model from the checkpoint
                vs.load(args.restore_from.as_ref().unwrap()).unwrap();
                println!("[+] Restored the model from the checkpoint")
            }

            // freeze?

            println!("[.] Next token probabilities for: [{}]...", prompt);
            let gen = next_token(device, model, tokenizer, prompt);
            println!("[+] distribution: {:#?}", gen);
        }
        Action::Generate {
            model,
            max_len,
            prompt,
        } => {
            // Build the model
            let (model, tokenizer) = create_model(&mut vs, model);
            println!("[+] Got a model and a tokenizer");

            // Restore the model from the checkpoint
            if args.restore_from.is_some() {
                println!("[.] Restoring the model from the checkpoint");
                // Restore the model from the checkpoint
                vs.load(args.restore_from.as_ref().unwrap()).unwrap();
                println!("[+] Restored the model from the checkpoint")
            }

            // freeze?

            println!("[.] Generating text: [{}]...", prompt);
            let gen = generate(device, model, tokenizer, prompt, max_len);
            println!("[+] Generated text: {}", gen);
        }
        Action::Train {
            model,
            training_params,
        } => {
            let model_name = match &model {
                Model::NanoGpt { .. } => "nanoGPT".to_string(),
                Model::Bigram { .. } => "bigram".to_string(),
                Model::GPT2 { args } => args.size.to_string(),
            };

            // Build the model
            let (model, tokenizer) = create_model(&mut vs, model);
            println!("[+] Got a model and a tokenizer");

            // Restore the model from the checkpoint
            if args.restore_from.is_some() {
                println!("[.] Restoring the model from the checkpoint");
                // Restore the model from the checkpoint
                vs.load(args.restore_from.as_ref().unwrap()).unwrap();
                println!("[+] Restored the model from the checkpoint")
            }

            let run_name = training_params.run_name.clone().unwrap_or_else(|| {
                let now: DateTime<Utc> = Utc::now();
                format!("{}_{}", model_name, now.format("%Y%m%d_%H%M%S"))
            });

            // only allow [a-zA-Z0-9_-] in the run name
            let run_name = run_name
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect::<String>();

            println!("[.] Training the model - ®[{}]...", run_name);
            train(vs, device, run_name, model, tokenizer, training_params);
            println!("[+] Trained the model");
        }
    }
}

/// Next token probabilities
fn next_token(
    device: tch::Device,
    model: Box<dyn LMModel>,
    tokenizer: Box<dyn Tokenizer>,
    xs: String,
) -> Vec<(String, f64)> {
    // generate some text
    let xs = tokenizer.encode(&xs);
    let xs = Tensor::from_slice(&xs).reshape([1, -1]).to(device);
    let probs = model.probabilities(&xs);
    let probs: Vec<f64> = probs.to(tch::Device::Cpu).reshape(-1).try_into().unwrap();

    let vocab = tokenizer.vocab();

    // zip the vocab and the probs
    vocab.into_iter().zip(probs.into_iter()).collect::<Vec<_>>()
}

/// Generate text
fn generate(
    device: tch::Device,
    model: Box<dyn LMModel>,
    tokenizer: Box<dyn Tokenizer>,
    xs: String,
    max_len: usize,
) -> String {
    // generate some text
    let xs = tokenizer.encode(&xs);
    let xs = Tensor::from_slice(&xs).reshape([1, -1]).to(device);
    let ys = model.generate(xs, max_len as i64);

    // decode the generated sequence of tokens
    let ys: Vec<i64> = ys.reshape(-1).try_into().unwrap();
    tokenizer.decode(&ys)
}

/// Train the model
fn train(
    mut vs: VarStore,
    device: tch::Device,
    run_name: String,
    model: Box<dyn LMModel>,
    tokenizer: Box<dyn Tokenizer>,
    training_params: TrainingParameters,
) {
    // Load the data
    let data = load_file(&training_params.dataset_path);
    println!("data.len(): {}", data.len());

    // print the first 200 characters
    println!("first 200 chars:\n----\n{}\n----\n", &data[0..1000]);

    // Split the data into training and validation sets
    let n = data.len() * 9 / 10;
    let train_data = &data[0..n];
    let valid_data = &data[n..];

    println!("train_data.len(): {}", train_data.len());
    println!("valid_data.len(): {}", valid_data.len());

    let kind = tch::Kind::Int;

    let train_data = TokenizedData::new(train_data, tokenizer.as_ref(), device, kind);
    let valid_data = TokenizedData::new(valid_data, tokenizer.as_ref(), device, kind);

    let block_size = model.block_size();
    let batch_size = training_params.batch_size;

    let mut train_dataloader = Loader::from_tokenized_data(train_data, block_size, batch_size);
    let mut valid_dataloader = Loader::from_tokenized_data(valid_data, block_size, batch_size);

    println!(
        "train_dataloader.n_batches(): {}",
        train_dataloader.n_batches()
    );
    println!(
        "valid_dataloader.n_batches(): {}",
        valid_dataloader.n_batches()
    );

    // Shuffle the batches
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(training_params.dataloader_rng_seed);

    train_dataloader.shuffle(&mut rng);
    valid_dataloader.shuffle(&mut rng);

    // print the names of the parameters
    vs.variables().iter().for_each(|(name, t)| {
        println!("{name}: {t:?}", name = name, t = t.size());
    });

    // number of parameters
    let nb_params = vs
        .trainable_variables()
        .iter()
        .map(|t| t.size().iter().product::<i64>())
        .sum::<i64>();
    println!("nb parameters: {}", nb_params);

    // let xs = Tensor::zeros([1, 1_i64], (tch::Kind::Int, device));
    // let max_len = 100;
    // let ys = model.generate(xs, max_len);
    //
    // // decode the generated sequence of tokens
    // let ys: Vec<i64> = Vec::<i64>::try_from(ys.reshape(-1)).unwrap();
    // let decoded = tokenizer.decode(&ys);
    // println!("decoded: {}", decoded);

    // FUTURE(ssoudan) support half precision training
    // vs.float();
    // vs.trainable_variables().iter().for_each(|t| {
    //     println!("t: {:?}", t);
    // });
    //
    // vs.variables().iter().for_each(|t| {
    //     println!("t: {:?}", t);
    // });

    // Initialize the progress bars
    let mut observer = Observer::default().with(Box::new(PbProgressReporter::default()));
    if let Some(tensorboard_dir) = training_params.tensorboard_dir {
        observer = observer.with(Box::new(TensorboardReporter::new(&tensorboard_dir)));
    }

    let learn_config = LearnConfig {
        n_epochs: training_params.n_epochs,
        lr: training_params.lr,
        steps_between_loss_estimation: training_params.steps_between_loss_estimation,
        loss_estimation_steps: training_params.loss_estimation_steps,
        max_grad_norm: if training_params.max_grad_norm > 0. {
            Some(training_params.max_grad_norm)
        } else {
            None
        },
    };

    learn(
        learn_config,
        run_name,
        &mut train_dataloader,
        &mut valid_dataloader,
        &mut rng,
        &mut vs,
        &model,
        &mut observer,
    );

    // save the model
    if let Some(final_checkpoint_path) = training_params.final_checkpoint_path {
        vs.save(final_checkpoint_path).unwrap();
    }

    if let Some(prompt) = training_params.prompt {
        let gen = generate(device, model, tokenizer, prompt.clone(), 500);
        println!("[i] after training: [{}]...{}", prompt, gen);
    }
}

fn create_model(vs: &mut VarStore, model: Model) -> (Box<dyn LMModel>, Box<dyn Tokenizer>) {
    match model {
        Model::NanoGpt {
            args:
                NanoGptArgs {
                    block_size,
                    n_embd,
                    n_layer,
                    n_head,
                    dropout,
                    bias,
                    tie_weights,
                    vocab_file,
                },
        } => {
            // Load the data
            let data = load_file(&vocab_file);
            // build the vocabulary
            let vocab = Vocab::new(&data);

            let config = NanoGptConfig {
                vocab_size: vocab.size() as i64,
                block_size,
                n_embd,
                n_head,
                n_layer,
                dropout,
                bias,
                tie_weights,
            };
            (
                Box::new(nanogptrs::model::NanoGpt::new(&vs.root(), config)),
                Box::new(vocab),
            )
        }
        Model::Bigram { vocab_file } => {
            // Load the data
            let data = load_file(&vocab_file);
            // build the vocabulary
            let vocab = Vocab::new(&data);
            (
                Box::new(nanogptrs::model::bigram::BigramLanguageModel::new(
                    &vs.root(),
                    vocab.size() as i64,
                )),
                Box::new(vocab),
            )
        }
        Model::GPT2 {
            args: GPT2Args {
                size,
                tokenizer_path,
            },
        } => {
            let dropout = 0.1;
            let bias = true; // GPT2 has bias

            let tokenizer = Gpt2Tokenizer::new(tokenizer_path);

            let config = match size {
                Gpt2Size::Normal => NanoGptConfig {
                    vocab_size: 50257,
                    block_size: 1024,
                    n_embd: 768,
                    n_head: 12,
                    n_layer: 12,
                    dropout,
                    bias,
                    tie_weights: true,
                },
                Gpt2Size::Medium => NanoGptConfig {
                    vocab_size: 50257,
                    block_size: 1024,
                    n_embd: 1024,
                    n_head: 16,
                    n_layer: 24,
                    dropout,
                    bias,
                    tie_weights: true,
                },
                Gpt2Size::Large => NanoGptConfig {
                    vocab_size: 50257,
                    block_size: 1024,
                    n_embd: 1280,
                    n_head: 20,
                    n_layer: 36,
                    dropout,
                    bias,
                    tie_weights: true,
                },
                Gpt2Size::XLarge => NanoGptConfig {
                    vocab_size: 50257,
                    block_size: 1024,
                    n_embd: 1600,
                    n_head: 25,
                    n_layer: 48,
                    dropout,
                    bias,
                    tie_weights: true,
                },
            };

            (
                Box::new(nanogptrs::model::NanoGpt::new(&vs.root(), config)),
                Box::new(tokenizer),
            )
        }
    }
}

struct LearnConfig {
    n_epochs: f32,
    lr: f64,
    steps_between_loss_estimation: usize,
    loss_estimation_steps: usize,
    max_grad_norm: Option<f64>,
}

#[allow(clippy::borrowed_box)]
fn learn<R: Rng>(
    learn_config: LearnConfig,
    run_name: String,
    train_dataloader: &mut Loader,
    valid_dataloader: &mut Loader,
    mut rng: &mut R,
    vs: &mut tch::nn::VarStore,
    model: &Box<dyn LMModel>,
    observer: &mut Observer,
) {
    let LearnConfig {
        n_epochs,
        lr,
        mut steps_between_loss_estimation,
        mut loss_estimation_steps,
        max_grad_norm,
    } = learn_config;

    // clones the dataloaders for the loss estimation
    let mut train_dataloader_loss = train_dataloader.clone();
    let mut valid_dataloader_loss = valid_dataloader.clone();

    let batches_per_epoch = train_dataloader.n_batches();

    let n_steps = (batches_per_epoch as f32 * n_epochs) as usize;
    let block_size = train_dataloader.block_size();
    let batch_size = train_dataloader.batch_size();

    // adjust the steps between loss estimation if necessary
    if n_steps <= steps_between_loss_estimation {
        println!(
            "[i] steps between loss estimation ({}) is larger than the number of steps ({}). Adjusting.",
            steps_between_loss_estimation, n_steps);
        steps_between_loss_estimation = n_steps / 5;
    }

    // adjust the loss estimation steps if necessary
    if loss_estimation_steps > steps_between_loss_estimation {
        println!(
            "[i] loss estimation steps ({}) is larger than the steps between loss estimation ({}). Adjusting.",
            loss_estimation_steps, steps_between_loss_estimation);
        loss_estimation_steps = steps_between_loss_estimation / 10;
    }

    observer.epoch_start(run_name, n_epochs, batches_per_epoch);

    train_dataloader.shuffle(&mut rng);

    // autocast(true, || {
    // train the model
    let mut opt = tch::nn::Adam::default().build(vs, lr).unwrap();

    observer.train_start(steps_between_loss_estimation);

    // work in terms of steps and not epochs
    for i in 0..n_steps {
        // get a batch - reshuffle if necessary
        let (xs, ys) = if let Some((xs, ys)) = train_dataloader.next_batch() {
            (xs, ys)
        } else {
            train_dataloader.shuffle(&mut rng);
            train_dataloader.next_batch().unwrap()
        };

        let logits = model.forward_t(&xs, true);

        let loss_value = loss(&logits, &ys);

        // opt.backward_step(&loss);
        opt.zero_grad();
        loss_value.backward();
        if let Some(max_grad_norm) = max_grad_norm {
            opt.clip_grad_norm(max_grad_norm);
        }
        opt.step();

        if i % 10 == 0 {
            observer.train_progress(i % steps_between_loss_estimation);
            observer.epoch_progress(i);
        }

        if (i % steps_between_loss_estimation == 0 && i != 0) || i == n_steps - 1 {
            observer.train_end();

            // loss estimation pb_reporter.estimate_start();
            observer.estimate_start();

            // Reshuffle the batches
            train_dataloader_loss.shuffle(&mut rng);
            valid_dataloader_loss.shuffle(&mut rng);

            let mut estimator =
                LossEstimator::new(&mut train_dataloader_loss, &mut valid_dataloader_loss, loss);
            let loss_estimates = estimator.estimate_loss(
                model.as_ref(),
                loss_estimation_steps, /* use the same number of batches for training and
                                        * validation */
                loss_estimation_steps,
                i * block_size * batch_size, // the current token
                observer,
            );

            observer.estimate_end(loss_estimates);

            if i != n_steps - 1 {
                observer.train_start(steps_between_loss_estimation);
            }
        }
    }
    // });

    observer.epoch_end();
}
