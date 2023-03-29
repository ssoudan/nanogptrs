//! NanoGPTRS: A rust implementation of the NanoGPT
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nanogptrs::data::{load_file, Loader, TokenizedData, Vocab};
use nanogptrs::learn;
use nanogptrs::learn::{estimate_loss, LossEstimates};
use nanogptrs::model::loss;
use rand_chacha::rand_core::SeedableRng;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::Tensor;

// TODO(ssoudan): Tensorboard?

/// Progress reporter that uses the `indicatif` crate to display progress bars.
struct PbProgressReporter {
    mb: MultiProgress,
    epoch_bar: Option<ProgressBar>,
    train_bar: Option<ProgressBar>,
    estimate_bar: Option<ProgressBar>,
    estimate_train_bar: Option<ProgressBar>,
    estimate_valid_bar: Option<ProgressBar>,
    current_epoch: usize,
    train_loss: f64,
    valid_loss: f64,
}

impl PbProgressReporter {
    /// Create a new progress reporter that uses the `indicatif` crate to display progress bars.
    pub fn new() -> Self {
        let mb = MultiProgress::new();
        PbProgressReporter {
            mb,
            epoch_bar: None,
            train_bar: None,
            estimate_bar: None,
            estimate_train_bar: None,
            estimate_valid_bar: None,
            current_epoch: 0,
            train_loss: 0.0,
            valid_loss: 0.0,
        }
    }
}

impl ProgressReporter for PbProgressReporter {
    fn epoch_start(&mut self, n_epochs: usize) {
        let epoch_bar = self.mb.add(ProgressBar::new(n_epochs as u64));
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} MASTER   [{elapsed_precise}] {bar:40.cyan/blue} Epoch {pos:>7}/{len:7} {msg}")
                .unwrap().progress_chars("##-"),
        );
        epoch_bar.tick();
        self.epoch_bar = Some(epoch_bar);
        self.current_epoch = 0;
    }
    fn epoch_progress(&mut self, current_epoch: usize) {
        if let Some(epoch_bar) = &self.epoch_bar {
            epoch_bar.set_position(current_epoch as u64);
        }
        self.current_epoch = current_epoch;
    }
    fn epoch_end(&mut self) {
        if let Some(epoch_bar) = &self.epoch_bar {
            epoch_bar.finish_and_clear();
        }
        self.epoch_bar = None;
    }
    fn train_start(&mut self, n_train_batches: usize) {
        let train_bar = self.mb.add(ProgressBar::new(n_train_batches as u64));
        train_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} TRAINING [{elapsed_precise}] {bar:20.green/blue} {pos:>7}/{len:7} {msg}")
                .unwrap().progress_chars("##-"),
        );
        train_bar.set_message(format!("Epoch {}", self.current_epoch));
        self.train_bar = Some(train_bar);
    }
    fn train_progress(&mut self, current_train_batches: usize) {
        if let Some(train_bar) = &self.train_bar {
            train_bar.set_position(current_train_batches as u64);
        }
    }
    fn train_end(&mut self) {
        if let Some(train_bar) = &self.train_bar {
            train_bar.finish();
        }
        self.train_bar = None;
    }
    fn estimate_start(&mut self) {
        let estimate_bar = self.mb.add(ProgressBar::new(2));
        estimate_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} EVAL     [{elapsed_precise}] {bar:20.magenta/blue} {pos:>7}/{len:7} {msg}")
                .unwrap().progress_chars("##-"),
        );
        estimate_bar.set_message(format!("Estimating epoch {}", self.current_epoch));
        self.estimate_bar = Some(estimate_bar);
    }
    fn estimate_progress(&mut self) {
        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.inc(1);
        }
    }
    fn estimate_end(&mut self, loss_estimates: LossEstimates) {
        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.set_message(format!(
                "Epoch {} train loss: {:.4}, valid loss: {:.4}",
                self.current_epoch, loss_estimates.train_loss, loss_estimates.valid_loss
            ));
            estimate_bar.finish();
        }
        self.estimate_bar = None;
    }
}

/// A trait for reporting progress during training.
pub trait ProgressReporter {
    /// Called before epoch starts.
    fn epoch_start(&mut self, n_epochs: usize);
    /// Called when an epoch ends.
    fn epoch_progress(&mut self, current_epoch: usize);
    /// Called when all epochs have been processed.
    fn epoch_end(&mut self);

    /// Called when the training starts.
    fn train_start(&mut self, n_train_batches: usize);
    /// Called when some batches have been processed.
    fn train_progress(&mut self, current_train_batches: usize);
    /// Called when the training ends.
    fn train_end(&mut self);

    /// Called when the loss estimation starts.
    fn estimate_start(&mut self);
    /// Called when a stage of the loss estimation are completed.
    fn estimate_progress(&mut self);
    /// Called when the loss estimation ends.
    fn estimate_end(&mut self, loss_estimates: LossEstimates);
}

impl learn::ProgressReporter for PbProgressReporter {
    fn train_loss_start(&mut self, total_train_batches: usize) {
        let train_loss_bar = self.mb.add(ProgressBar::new(total_train_batches as u64));
        train_loss_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green}        T [{elapsed_precise}] {bar:20.yellow/blue} {pos:>7}/{len:7} {msg}")
                .unwrap().progress_chars("##-"),
        );

        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.set_message("Train loss estimation");
        }
        self.estimate_train_bar = Some(train_loss_bar);
    }

    fn train_loss_progress(&mut self, current_train_batches: usize) {
        if let Some(train_loss_bar) = &self.estimate_train_bar {
            train_loss_bar.set_position(current_train_batches as u64);
        }
    }

    fn train_loss_end(&mut self, train_loss: f64) {
        if let Some(train_loss_bar) = &self.estimate_train_bar {
            train_loss_bar.finish_and_clear();
        }
        self.estimate_train_bar = None;

        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.set_message(format!(
                "Epoch {} Train loss: {}",
                self.current_epoch, train_loss
            ));
        }

        self.train_loss = train_loss;

        // progress on the estimate bar
        self.estimate_progress();
    }

    fn valid_loss_start(&mut self, total_valid_batches: usize) {
        let valid_loss_bar = self.mb.add(ProgressBar::new(total_valid_batches as u64));
        valid_loss_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green}        E [{elapsed_precise}] {bar:20.yellow/blue} {pos:>7}/{len:7} {msg}")
                .unwrap().progress_chars("##-"),
        );
        self.estimate_valid_bar = Some(valid_loss_bar);
    }

    fn valid_loss_progress(&mut self, current_valid_batches: usize) {
        if let Some(valid_loss_bar) = &self.estimate_valid_bar {
            valid_loss_bar.set_position(current_valid_batches as u64);
        }
    }

    fn valid_loss_end(&mut self, valid_loss: f64) {
        if let Some(valid_loss_bar) = &self.estimate_valid_bar {
            valid_loss_bar.finish_and_clear();
        }
        self.estimate_valid_bar = None;

        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.set_message(format!(
                "Epoch {} Train loss: {} Valid loss: {}",
                self.current_epoch, self.train_loss, valid_loss
            ));
        }

        self.valid_loss = valid_loss;
    }
}

fn main() {
    let device = tch::Device::Cpu;

    println!("Hello, world!");

    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();

    let data = load_file();
    println!("data.len(): {}", data.len());

    // print the first 1000 characters
    println!("first 1000 chars: {}", &data[0..1000]);

    // build the vocabulary
    let vocab = Vocab::new(&data);
    println!("vocab.len(): {}", vocab.size());
    println!("vocab: {:?}", vocab.chars());

    // encode the first 1000 characters
    let encoded = vocab.encode(&data[0..1000]);
    println!("encoded: {:?}", encoded);

    // decode the first 1000 characters
    let decoded = vocab.decode(&encoded);
    println!("decoded: {}", decoded);

    // check that the decoded string is the same as the original
    assert_eq!(decoded, &data[0..1000]);

    // Split the data into training and validation sets
    let n = data.len() * 9 / 10;
    let train_data = &data[0..n];
    let valid_data = &data[n..];

    println!("train_data.len(): {}", train_data.len());
    println!("valid_data.len(): {}", valid_data.len());

    let train_data = TokenizedData::new(train_data, vocab.clone());
    let valid_data = TokenizedData::new(valid_data, vocab.clone());

    let batch_size = 4;
    let seq_len = 8;

    let mut train_dataloader = Loader::from_tokenized_data(train_data, seq_len, batch_size);
    let mut valid_dataloader = Loader::from_tokenized_data(valid_data, seq_len, batch_size);

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

    let (samples, targets) = train_dataloader.next_batch().unwrap();
    println!("samples: {:?}", samples);
    println!("targets: {:?}", targets);

    ///////

    let vs = tch::nn::VarStore::new(device);

    let model = nanogptrs::model::BigramLanguageModel::new(&vs.root(), vocab.size() as i64);

    let xs = Tensor::zeros(&[1, 1], (tch::Kind::Int64, device));
    let max_len = 100;
    let ys = model.generate(xs, max_len);
    println!("generated: {:?}", ys);

    // decode the generated sequence of tokens
    let ys: Vec<i64> = ys.into();
    let decoded = vocab.decode(&ys);
    println!("decoded: {}", decoded);

    // train the model
    let lr = 1e-3;
    let n_epochs = 10;

    let mut opt = tch::nn::Adam::default().build(&vs, lr).unwrap();

    // Initialize the progress bars
    let mut pb_reporter = PbProgressReporter::new();

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
            //     .view([batch_size as i64, seq_len as i64]);
            // let ys = Tensor::of_slice(&ys)
            //     .to_kind(tch::Kind::Int64)
            //     .to(device)
            //     .view([batch_size as i64, seq_len as i64]);

            let logits = model.forward_t(&xs, true);

            let loss = loss(&logits, &ys);

            // opt.backward_step(&loss);
            opt.zero_grad();
            loss.backward();
            opt.step();

            train_n += 1;

            if train_n % 1000 == 0 {
                pb_reporter.train_progress(train_n);
            }
        }

        pb_reporter.train_end();

        // loss estimation
        pb_reporter.estimate_start();

        // Reshuffle the batches
        train_dataloader.shuffle(&mut rng);
        valid_dataloader.shuffle(&mut rng);

        let iters = valid_dataloader.n_batches();
        let loss_estimates = estimate_loss(
            &mut train_dataloader,
            &mut valid_dataloader,
            &model,
            device,
            iters, // use the same number of batches for training and validation
            iters,
            &mut pb_reporter,
            loss,
        );

        pb_reporter.estimate_end(loss_estimates);

        pb_reporter.epoch_progress(epoch + 1);
    }
    pb_reporter.epoch_end();

    // generate some text
    let xs = Tensor::zeros(&[1, 1], (tch::Kind::Int64, tch::Device::Cpu));
    let max_len = 100;
    let ys = model.generate(xs, max_len);
    println!("generated: {:?}", ys);

    // decode the generated sequence of tokens
    let ys: Vec<i64> = ys.into();
    let decoded = vocab.decode(&ys);
    println!("decoded: {}", decoded);
}
