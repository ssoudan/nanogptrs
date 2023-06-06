/// Logger for training.
pub mod logger;

use std::fmt::Write;

use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};

use crate::estimate;
use crate::estimate::LossEstimates;

/// A trait for reporting progress during training.
#[allow(unused_variables)]
pub trait ProgressReporter: estimate::ProgressReporter {
    /// Called before epoch starts.
    fn epoch_start(&mut self, run_name: String, n_epochs: f32, batches_per_epochs: usize) {}
    /// Called when an epoch ends.
    fn epoch_progress(&mut self, current_epoch: usize) {}
    /// Called when all epochs have been processed.
    fn epoch_end(&mut self) {}

    /// Called when the training starts.
    fn train_start(&mut self, n_train_batches: usize) {}
    /// Called when some batches have been processed.
    fn train_progress(&mut self, current_train_batches: usize) {}
    /// Called when the training ends.
    fn train_end(&mut self) {}

    /// Called when the loss estimation starts.
    fn estimate_start(&mut self) {}
    /// Called when a stage of the loss estimation are completed.
    fn estimate_progress(&mut self) {}
    /// Called when the loss estimation ends.
    fn estimate_end(&mut self, loss_estimates: LossEstimates) {}
}

/// Training observer.
pub struct Observer {
    reporters: Vec<Box<dyn ProgressReporter>>,
}

impl Observer {
    /// Add a reporter to the observer.
    pub fn with(mut self, reporter: Box<dyn ProgressReporter>) -> Self {
        self.reporters.push(reporter);
        self
    }

    /// Build the observer.
    pub fn build(self) -> Self {
        self
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self { reporters: vec![] }
    }
}

impl ProgressReporter for Observer {
    fn epoch_start(&mut self, run_name: String, n_epochs: f32, batches_per_epochs: usize) {
        for reporter in &mut self.reporters {
            reporter.epoch_start(run_name.clone(), n_epochs, batches_per_epochs);
        }
    }

    fn epoch_progress(&mut self, current_epoch: usize) {
        for reporter in &mut self.reporters {
            reporter.epoch_progress(current_epoch);
        }
    }

    fn epoch_end(&mut self) {
        for reporter in &mut self.reporters {
            reporter.epoch_end();
        }
    }

    fn train_start(&mut self, n_train_batches: usize) {
        for reporter in &mut self.reporters {
            reporter.train_start(n_train_batches);
        }
    }

    fn train_progress(&mut self, current_train_batches: usize) {
        for reporter in &mut self.reporters {
            reporter.train_progress(current_train_batches);
        }
    }

    fn train_end(&mut self) {
        for reporter in &mut self.reporters {
            reporter.train_end();
        }
    }

    fn estimate_start(&mut self) {
        for reporter in &mut self.reporters {
            reporter.estimate_start();
        }
    }

    fn estimate_progress(&mut self) {
        for reporter in &mut self.reporters {
            reporter.estimate_progress();
        }
    }

    fn estimate_end(&mut self, loss_estimates: LossEstimates) {
        for reporter in &mut self.reporters {
            reporter.estimate_end(loss_estimates.clone());
        }
    }
}

impl estimate::ProgressReporter for Observer {
    fn train_loss_start(&mut self, total_train_batches: usize) {
        for reporter in &mut self.reporters {
            reporter.train_loss_start(total_train_batches);
        }
    }

    fn train_loss_progress(&mut self, current_train_batches: usize) {
        for reporter in &mut self.reporters {
            reporter.train_loss_progress(current_train_batches);
        }
    }

    fn train_loss_end(&mut self, train_loss: f64) {
        for reporter in &mut self.reporters {
            reporter.train_loss_end(train_loss);
        }
    }

    fn valid_loss_start(&mut self, total_valid_batches: usize) {
        for reporter in &mut self.reporters {
            reporter.valid_loss_start(total_valid_batches);
        }
    }

    fn valid_loss_progress(&mut self, current_valid_batches: usize) {
        for reporter in &mut self.reporters {
            reporter.valid_loss_progress(current_valid_batches);
        }
    }

    fn valid_loss_end(&mut self, valid_loss: f64) {
        for reporter in &mut self.reporters {
            reporter.valid_loss_end(valid_loss);
        }
    }
}

/// Progress reporter that uses the `indicatif` crate to display progress bars.
///
/// We want progress bar like: ```[198/200 35:16 < 00:21, 0.09/s]```
pub struct PbProgressReporter {
    mb: MultiProgress,
    epoch_bar: Option<ProgressBar>,
    train_bar: Option<ProgressBar>,
    estimate_bar: Option<ProgressBar>,
    estimate_train_bar: Option<ProgressBar>,
    estimate_valid_bar: Option<ProgressBar>,
    current_epoch: f32,
    batches_per_epoch: usize,
    train_loss: f64,
    valid_loss: f64,
}

impl Default for PbProgressReporter {
    fn default() -> Self {
        let mb = MultiProgress::new();
        PbProgressReporter {
            mb,
            epoch_bar: None,
            train_bar: None,
            estimate_bar: None,
            estimate_train_bar: None,
            estimate_valid_bar: None,
            current_epoch: 0.,
            batches_per_epoch: 0,
            train_loss: 0.0,
            valid_loss: 0.0,
        }
    }
}

impl ProgressReporter for PbProgressReporter {
    fn epoch_start(&mut self, _run_name: String, n_epochs: f32, batches_per_epochs: usize) {
        let epoch_bar = self.mb.add(ProgressBar::new(
            (n_epochs * batches_per_epochs as f32) as u64,
        ));
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} MASTER   {bar:20.cyan/blue} [{pos:>7}/{len:7} {elapsed_precise} < {eta_precise}, {per_sec_short:.2}, Epoch {epoch_progress}] {msg}")
                .unwrap()
                .with_key("per_sec_short", 
                          |state: &ProgressState, w: &mut dyn Write|
                              write!(w, "{:>7.1}/s", state.per_sec()).unwrap())
                .with_key("epoch_progress", 
                          move |state: &ProgressState, w: &mut dyn Write|
                              write!(w, "{:>4.1}/{}",  state.pos() as f32/batches_per_epochs as f32, n_epochs).unwrap()
                          )
                .progress_chars("##-"),
        );

        epoch_bar.tick();
        self.epoch_bar = Some(epoch_bar);
        self.current_epoch = 0.;
        self.batches_per_epoch = batches_per_epochs;
    }

    fn epoch_progress(&mut self, current_batch: usize) {
        if let Some(epoch_bar) = &self.epoch_bar {
            epoch_bar.set_position(current_batch as u64);
        }

        self.current_epoch = current_batch as f32 / self.batches_per_epoch as f32;
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

                .template("{spinner:.green} TRAINING {bar:20.green/blue} [{pos:>7}/{len:7} {elapsed_precise} < {eta_precise}, {per_sec_short:.2}] {msg}")
                .unwrap()
                .with_key("per_sec_short", 
                                   |state: &ProgressState, w: &mut dyn Write|
                                       write!(w, "{:>7.1}/s", state.per_sec()).unwrap())
                .progress_chars("##-"));
        train_bar.tick();
        self.train_bar = Some(train_bar);
    }

    fn train_progress(&mut self, current_train_batches: usize) {
        if let Some(train_bar) = &self.train_bar {
            train_bar.set_position(current_train_batches as u64);
        }
    }

    fn train_end(&mut self) {
        if let Some(train_bar) = &self.train_bar {
            train_bar.finish_and_clear();
        }
        self.train_bar = None;
    }

    fn estimate_start(&mut self) {
        let estimate_bar = self.mb.add(ProgressBar::new(2));
        estimate_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} EVAL     {bar:20.magenta/blue} [{pos:>7}/{len:7} {elapsed_precise} < {eta_precise}, {per_sec_short:.2}] {msg}")
                .unwrap().with_key("per_sec_short",
                                   |state: &ProgressState, w: &mut dyn Write|
                                       write!(w, "{:>7.1}/s", state.per_sec()).unwrap())
                .progress_chars("##-"));
        estimate_bar.tick();
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
                "Epoch {:>4.1}, Train loss: {:.4}, Valid loss: {:.4}",
                self.current_epoch, loss_estimates.train_loss, loss_estimates.valid_loss
            ));
            estimate_bar.finish(); // keep the bar
        }
        self.estimate_bar = None;
    }
}

impl estimate::ProgressReporter for PbProgressReporter {
    fn train_loss_start(&mut self, total_train_batches: usize) {
        let train_loss_bar = self.mb.add(ProgressBar::new(total_train_batches as u64));
        train_loss_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green}        T {bar:20.yellow/blue} [{pos:>7}/{len:7} {elapsed_precise} < {eta_precise}, {per_sec_short:.2}] {msg}")
                .unwrap().with_key("per_sec_short",
                                   |state: &ProgressState, w: &mut dyn Write|
                                       write!(w, "{:>7.1}/s", state.per_sec()).unwrap())
                .progress_chars("##-"));

        self.estimate_train_bar = Some(train_loss_bar);
    }

    fn train_loss_progress(&mut self, current_train_batches: usize) {
        if let Some(estimate_train_bar) = &self.estimate_train_bar {
            estimate_train_bar.set_position(current_train_batches as u64);
        }
    }

    fn train_loss_end(&mut self, train_loss: f64) {
        if let Some(estimate_train_bar) = &self.estimate_train_bar {
            estimate_train_bar.finish_and_clear();
        }
        self.estimate_train_bar = None;

        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.set_message(format!(
                "Epoch {:>4.1}, Train loss: {:.4}",
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
                .template("{spinner:.green}        E {bar:20.yellow/blue} [{pos:>7}/{len:7} {elapsed_precise} < {eta_precise}, {per_sec_short:.2}] {msg}")
                .unwrap().with_key("per_sec_short",
                                   |state: &ProgressState, w: &mut dyn Write|
                                       write!(w, "{:>7.1}/s", state.per_sec()).unwrap())
                .progress_chars("##-"));
        self.estimate_valid_bar = Some(valid_loss_bar);
    }

    fn valid_loss_progress(&mut self, current_valid_batches: usize) {
        if let Some(estimate_valid_bar) = &self.estimate_valid_bar {
            estimate_valid_bar.set_position(current_valid_batches as u64);
        }
    }

    fn valid_loss_end(&mut self, valid_loss: f64) {
        if let Some(estimate_valid_bar) = &self.estimate_valid_bar {
            estimate_valid_bar.finish_and_clear();
        }
        self.estimate_valid_bar = None;

        if let Some(estimate_bar) = &self.estimate_bar {
            estimate_bar.set_message(format!(
                "Epoch {:>4.1}, Train loss: {:.4}, valid loss: {:.4}",
                self.current_epoch, self.train_loss, valid_loss
            ));
        }

        self.valid_loss = valid_loss;
    }
}
