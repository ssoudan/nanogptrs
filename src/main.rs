//! NanoGPTRS: A rust implementation of the NanoGPT
use clap::{Parser, Subcommand, ValueEnum};
use nanogptrs::data::{load_file, Loader, TokenizedData, Vocab};
use nanogptrs::estimate::LossEstimator;
use nanogptrs::learn::{PbProgressReporter, ProgressReporter};
use nanogptrs::model::{loss, LMModel};
use rand_chacha::rand_core::SeedableRng;
use tch::nn::OptimizerConfig;
use tch::Tensor;

// TODO(ssoudan): Tensorboard?
// TODO(ssoudan): FP precision
// TODO(ssoudan): Count the number of parameters

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
    // #[arg(short, long, default_value = "0.001")]
    #[arg(long, default_value = "0.0003")]
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

    let train_data = TokenizedData::new(train_data, vocab.clone(), device);
    let valid_data = TokenizedData::new(valid_data, vocab.clone(), device);

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

    let vs = tch::nn::VarStore::new(device);

    let model: Box<dyn LMModel> = match args.model.unwrap_or_default() {
        Model::NanoGpt(NanoGptArgs {
            n_embd,
            n_layer,
            n_head,
        }) => Box::new(nanogptrs::model::NanoGpt::new(
            &vs.root(),
            vocab.size() as i64,
            block_size as i64,
            n_embd,
            n_head,
            n_layer,
        )),
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

    // train the model
    let mut opt = tch::nn::Adam::default().build(&vs, lr).unwrap();

    // half precision training
    // TODO(ssoudan) support half precision training
    // vs.half();
    // vs.trainable_variables().iter().for_each(|t| {
    //     println!("t: {:?}", t);
    // });
    //
    // vs.variables().iter().for_each(|t| {
    //     println!("t: {:?}", t);
    // });

    // Initialize the progress bars
    let mut pb_reporter = PbProgressReporter::default();

    pb_reporter.epoch_start(n_epochs);
    for epoch in 0..n_epochs {
        let mut train_n = 0;

        pb_reporter.train_start(train_dataloader.n_batches());

        train_dataloader.shuffle(&mut rng);
        while let Some((xs, ys)) = train_dataloader.next_batch() {
            // let xs: Vec<i64> = xs.into_iter().flatten().collect();
            // let ys: Vec<i64> = ys.into_iter().flatten().collect();
            //
            // let xs = Tensor::of_slice(&xs)
            //     .to_kind(tch::Kind::Int64)
            //     .to(device)
            //     .view([batch_size as i64, block_size as i64]);
            // let ys = Tensor::of_slice(&ys)
            //     .to_kind(tch::Kind::Int64)
            //     .to(device)
            //     .view([batch_size as i64, block_size as i64]);

            let logits = model.forward_t(&xs, true);

            let loss = loss(&logits, &ys);

            // opt.backward_step(&loss);
            opt.zero_grad();
            loss.backward();
            opt.step();

            train_n += 1;

            if train_n % 10 == 0 {
                pb_reporter.train_progress(train_n);
            }
        }

        pb_reporter.train_end();

        // loss estimation pb_reporter.estimate_start();

        // Reshuffle the batches
        train_dataloader.shuffle(&mut rng);
        valid_dataloader.shuffle(&mut rng);

        let iters = valid_dataloader.n_batches();
        let mut estimator = LossEstimator::new(&mut train_dataloader, &mut valid_dataloader, loss);
        let loss_estimates = estimator.estimate_loss(
            model.as_ref(),
            iters, // use the same number of batches for training and validation
            iters,
            &mut pb_reporter,
        );

        pb_reporter.estimate_end(loss_estimates);

        pb_reporter.epoch_progress(epoch + 1);
    }
    pb_reporter.epoch_end();

    // generate some text
    let xs = Tensor::zeros(&[1, block_size as i64], (tch::Kind::Int64, device));
    let max_len = 1500;
    let ys = model.generate(xs, max_len);

    // decode the generated sequence of tokens
    let ys: Vec<i64> = ys.into();
    let decoded = vocab.decode(&ys);
    println!("decoded: {}", decoded);
}
