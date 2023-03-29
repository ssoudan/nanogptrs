use crate::estimate;
use crate::estimate::LossEstimates;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Progress reporter that uses the `indicatif` crate to display progress bars.
pub struct PbProgressReporter {
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
                .template("{spinner:.green} TRAINING [{elapsed_precise}] {bar:20.green/blue} {pos:>7}/{len:7} {per_sec:.2} {msg}")
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
                .template("{spinner:.green} EVAL     [{elapsed_precise}] {bar:20.magenta/blue} {pos:>7}/{len:7} {per_sec:.2} {msg}")
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
                "Epoch {} Train loss: {:.4}, Valid loss: {:.4}",
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

impl estimate::ProgressReporter for PbProgressReporter {
    fn train_loss_start(&mut self, total_train_batches: usize) {
        let train_loss_bar = self.mb.add(ProgressBar::new(total_train_batches as u64));
        train_loss_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green}        T [{elapsed_precise}] {bar:20.yellow/blue} {pos:>7}/{len:7} {per_sec:.2} {msg}")
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
                "Epoch {} Train loss: {:.4}",
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
                .template("{spinner:.green}        E [{elapsed_precise}] {bar:20.yellow/blue} {pos:>7}/{len:7} {per_sec:.2} {msg}")
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
                "Epoch {} Train loss: {:.4}, valid loss: {:.4}",
                self.current_epoch, self.train_loss, valid_loss
            ));
        }

        self.valid_loss = valid_loss;
    }
}
