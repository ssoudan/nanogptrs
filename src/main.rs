//! NanoGPTRS: A rust implementation of the NanoGPT
use clap::{Parser, Subcommand, ValueEnum};
use nanogptrs::data::{load_file, Loader, TokenizedData, Vocab};
use nanogptrs::estimate::LossEstimator;
use nanogptrs::learn::{PbProgressReporter, ProgressReporter};
use nanogptrs::model::{loss, LMModel, NanoGptConfig};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use tch::nn::OptimizerConfig;
use tch::{autocast, Tensor};

// TODO(ssoudan): Tensorboard?

/// Torch device to use.
#[derive(ValueEnum, Debug, Clone, Copy, Default)]
enum Device {
    /// CPU
    #[default]
    Cpu,
    /// CUDA if available
    Cuda,
    /// MPS
    Mps,
}

/// Arguments for the NanoGPT model.
#[derive(Parser, Debug, Clone, Copy)]
struct NanoGptArgs {
    /// the size of the embedding to use
    #[arg(short = 'E', long, default_value = "384")]
    n_embd: i64,
    /// Number of layers
    #[arg(short = 'L', long, default_value = "6")]
    n_layer: i64,
    /// Number of heads
    #[arg(short = 'H', long, default_value = "6")]
    n_head: i64,
}

impl Default for NanoGptArgs {
    fn default() -> Self {
        Self {
            n_embd: 384,
            n_layer: 6,
            n_head: 6,
        }
    }
}

/// Training parameters.
#[derive(Parser, Debug, Clone, Copy)]
struct TrainingParameters {
    /// Learning rate
    // #[arg(short, long, default_value = "0.0001")]
    #[arg(long, default_value = "0.0001")]
    lr: f64,
    /// Number of epochs
    #[arg(long, default_value = "1")]
    n_epochs: usize,
    /// Batch size
    // #[arg(short, long, default_value = "32")]
    #[arg(long, default_value = "64")]
    batch_size: usize,
    /// Maximum sequence length
    #[arg(long, default_value = "256")]
    block_size: usize,
    /// Number of training iterations before evaluating the losses on the
    /// training and validation sets.
    #[arg(long, default_value = "750")]
    steps_between_loss_estimation: usize,
    /// Number of batches to use for estimating the losses.
    #[arg(long, default_value = "150")]
    loss_estimation_steps: usize,
}

/// The model to use.
#[derive(Subcommand, Debug, Clone, Copy)]
enum Model {
    /// NanoGPT
    NanoGpt(NanoGptArgs),
    /// BigramLanguageModel
    BigramLanguageModel,
}

impl Default for Model {
    fn default() -> Self {
        Self::NanoGpt(NanoGptArgs::default())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The device to use
    #[arg(short, long, default_value = "cuda")]
    device: Device,

    /// The model to use
    #[command(subcommand)]
    model: Option<Model>,

    /// Training parameters
    #[clap(flatten)]
    training_params: TrainingParameters,
}

fn main() {
    let args = Args::parse();

    let device = match args.device {
        Device::Cpu => tch::Device::Cpu,
        Device::Cuda => tch::Device::cuda_if_available(),
        Device::Mps => tch::Device::Mps,
    };
    let block_size = args.training_params.block_size;
    let batch_size = args.training_params.batch_size;
    let n_epochs = args.training_params.n_epochs;
    let lr = args.training_params.lr;

    // print a banner with nanoGPTrs
    println!(
        r#"
                                 ****   ****   *****                
                                *    *  *   *    *                  
                                *       *   *    *                  
* ***    ****   * ***    ****   *       *   *    *    * **   ****   
**   *       *  **   *  *    *  *       ****     *     *    *    *  
*    *   *****  *    *  *    *  *  ***  *        *     *     **     
*    *  *    *  *    *  *    *  *    *  *        *     *       **   
*    *  *   **  *    *  *    *  *   **  *        *     *    *    *  
*    *   *** *  *    *   ****    *** *  *        *     *     ****   
"#
    );

    // if not built in release mode, print a big warning
    #[cfg(debug_assertions)]
    {
        println!("WARNING: This is a debug build. It will be very slow.");
    }

    // Load the data
    let data = load_file();
    println!("data.len(): {}", data.len());

    // print the first 200 characters
    println!("first 200 chars:\n----\n{}\n----\n", &data[0..1000]);

    // build the vocabulary
    let vocab = Vocab::new(&data);
    println!("vocab.len(): {}", vocab.size());
    println!("vocab: {:?}", vocab.chars());

    // Split the data into training and validation sets
    let n = data.len() * 9 / 10;
    let train_data = &data[0..n];
    let valid_data = &data[n..];

    println!("train_data.len(): {}", train_data.len());
    println!("valid_data.len(): {}", valid_data.len());

    let kind = tch::Kind::Int;

    let train_data = TokenizedData::new(train_data, vocab.clone(), device, kind);
    let valid_data = TokenizedData::new(valid_data, vocab.clone(), device, kind);

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
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);

    train_dataloader.shuffle(&mut rng);
    valid_dataloader.shuffle(&mut rng);

    ///////

    let mut vs = tch::nn::VarStore::new(device);

    let model: Box<dyn LMModel> = match args.model.unwrap_or_default() {
        Model::NanoGpt(NanoGptArgs {
            n_embd,
            n_layer,
            n_head,
        }) => {
            let config = NanoGptConfig {
                vocab_size: vocab.size() as i64,
                block_size: block_size as i64,
                n_embd,
                n_head,
                n_layer,
            };
            Box::new(nanogptrs::model::NanoGpt::new(&vs.root(), config))
        }
        Model::BigramLanguageModel => Box::new(nanogptrs::model::BigramLanguageModel::new(
            &vs.root(),
            vocab.size() as i64,
        )),
    };

    // number of parameters
    let nb_params = vs
        .trainable_variables()
        .iter()
        .map(|t| t.size().iter().product::<i64>())
        .sum::<i64>();
    println!("nb parameters: {}", nb_params);

    let xs = Tensor::zeros(&[1, 1_i64], (tch::Kind::Int64, device));
    let max_len = 100;
    let ys = model.generate(xs, max_len);

    // decode the generated sequence of tokens
    let ys: Vec<i64> = ys.into();
    let decoded = vocab.decode(&ys);
    println!("decoded: {}", decoded);

    // half precision training
    // TODO(ssoudan) support half precision training
    vs.float();
    // vs.trainable_variables().iter().for_each(|t| {
    //     println!("t: {:?}", t);
    // });
    //
    // vs.variables().iter().for_each(|t| {
    //     println!("t: {:?}", t);
    // });

    // Initialize the progress bars
    let mut pb_reporter = PbProgressReporter::default();

    let learn_config = LearnConfig {
        n_epochs,
        lr,
        steps_between_loss_estimation: args.training_params.steps_between_loss_estimation,
        loss_estimation_steps: args.training_params.loss_estimation_steps,
    };

    learn(
        learn_config,
        &mut train_dataloader,
        &mut valid_dataloader,
        &mut rng,
        &mut vs,
        &model,
        &mut pb_reporter,
    );

    // generate some text
    let xs = Tensor::zeros(&[1, block_size as i64], (tch::Kind::Int64, device));
    let max_len = 1500;
    let ys = model.generate(xs, max_len);

    // decode the generated sequence of tokens
    let ys: Vec<i64> = ys.into();
    let decoded = vocab.decode(&ys);
    println!("decoded: {}", decoded);
}

struct LearnConfig {
    n_epochs: usize,
    lr: f64,
    steps_between_loss_estimation: usize,
    loss_estimation_steps: usize,
}

#[allow(clippy::borrowed_box)]
fn learn<R: Rng>(
    learn_config: LearnConfig,
    train_dataloader: &mut Loader,
    valid_dataloader: &mut Loader,
    mut rng: &mut R,
    vs: &mut tch::nn::VarStore,
    model: &Box<dyn LMModel>,
    pb_reporter: &mut PbProgressReporter,
) {
    let LearnConfig {
        n_epochs,
        lr,
        steps_between_loss_estimation,
        loss_estimation_steps,
    } = learn_config;

    // clones the dataloaders for the loss estimation
    let mut train_dataloader_loss = train_dataloader.clone();
    let mut valid_dataloader_loss = valid_dataloader.clone();

    let batches_per_epoch = train_dataloader.n_batches();

    pb_reporter.epoch_start(n_epochs, batches_per_epoch);

    let n_steps = batches_per_epoch * n_epochs;

    train_dataloader.shuffle(&mut rng);

    autocast(true, || {
        // train the model
        let mut opt = tch::nn::Adam::default().build(vs, lr).unwrap();

        pb_reporter.train_start(steps_between_loss_estimation);

        // work in term of steps and not epochs
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
            opt.step();

            if i % 10 == 0 {
                pb_reporter.train_progress(i % steps_between_loss_estimation);
                pb_reporter.epoch_progress(i);
            }

            if (i % steps_between_loss_estimation == 0 && i != 0) || i == n_steps - 1 {
                pb_reporter.train_end();

                // loss estimation pb_reporter.estimate_start();
                pb_reporter.estimate_start();

                // Reshuffle the batches
                train_dataloader_loss.shuffle(&mut rng);
                valid_dataloader_loss.shuffle(&mut rng);

                let mut estimator = LossEstimator::new(
                    &mut train_dataloader_loss,
                    &mut valid_dataloader_loss,
                    loss,
                );
                let loss_estimates = estimator.estimate_loss(
                    model.as_ref(),
                    loss_estimation_steps, /* use the same number of batches for training and
                                            * validation */
                    loss_estimation_steps,
                    pb_reporter,
                );

                pb_reporter.estimate_end(loss_estimates);

                if i != n_steps - 1 {
                    pb_reporter.train_start(steps_between_loss_estimation);
                }
            }
        }
    });

    pb_reporter.epoch_end();
}
