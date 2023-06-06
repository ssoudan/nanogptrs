use std::path::Path;

use tensorboard_rs::summary_writer::SummaryWriter;

use crate::estimate;
use crate::estimate::LossEstimates;
use crate::learn::ProgressReporter;

/// Reporter for Tensorboard
pub struct TensorboardReporter {
    writer: Option<SummaryWriter>,
    base_dir: String,
}

impl TensorboardReporter {
    /// Create a new TensorboardReporter
    pub fn new(base_dir: &str) -> Self {
        Self {
            writer: None,
            base_dir: base_dir.to_string(),
        }
    }
}

impl estimate::ProgressReporter for TensorboardReporter {}

impl ProgressReporter for TensorboardReporter {
    fn epoch_start(&mut self, run_name: String, n_epochs: f32, batches_per_epochs: usize) {
        let run_name = format!("run_{}_{}_{}", run_name, n_epochs, batches_per_epochs);
        let logdir = Path::new(&self.base_dir).join(run_name);
        // create a new writer
        self.writer = Some(SummaryWriter::new(&logdir));
    }

    fn epoch_end(&mut self) {
        // close the writer
        self.writer = None;
    }

    fn estimate_end(&mut self, loss_estimates: LossEstimates) {
        if let Some(writer) = &mut self.writer {
            writer.add_scalar(
                "train_loss",
                loss_estimates.train_loss as f32,
                loss_estimates.current_token,
            );
            writer.add_scalar(
                "valid_loss",
                loss_estimates.valid_loss as f32,
                loss_estimates.current_token,
            );
        }
    }
}
