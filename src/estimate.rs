use tch::Tensor;

use crate::data::Loader;
use crate::model::LMModel;

/// Estimates of the loss on the training and validation sets.
pub struct LossEstimates {
    /// Loss on the training set.
    pub train_loss: f64,
    /// Loss on the validation set.
    pub valid_loss: f64,
}

/// Interface for a progress reporter.
pub trait ProgressReporter {
    /// Called at the start of the loss estimation for the training sets.
    fn train_loss_start(&mut self, total_train_batches: usize);
    /// Called to update the progress of the loss estimation for the training
    /// sets.
    fn train_loss_progress(&mut self, current_train_batches: usize);
    /// Called at the end of the loss estimation for the training sets.
    fn train_loss_end(&mut self, train_loss: f64);
    /// Called at the start of the loss estimation for the validation sets.
    fn valid_loss_start(&mut self, total_valid_batches: usize);
    /// Called to update the progress of the loss estimation for the validation
    /// sets.
    fn valid_loss_progress(&mut self, current_valid_batches: usize);
    /// Called at the end of the loss estimation for the validation sets.
    fn valid_loss_end(&mut self, valid_loss: f64);
}

/// A progress reporter that does nothing.
pub struct NullProgressReporter;

impl ProgressReporter for NullProgressReporter {
    fn train_loss_start(&mut self, _total_train_batches: usize) {}

    fn train_loss_progress(&mut self, _current_train_batches: usize) {}

    fn train_loss_end(&mut self, _train_loss: f64) {}

    fn valid_loss_start(&mut self, _total_valid_batches: usize) {}

    fn valid_loss_progress(&mut self, _current_valid_batches: usize) {}

    fn valid_loss_end(&mut self, _valid_loss: f64) {}
}

/// Loss estimator.
pub struct LossEstimator<'a> {
    train_dataloader: &'a mut Loader,
    valid_dataloader: &'a mut Loader,
    loss: fn(&Tensor, &Tensor) -> Tensor,
}

impl<'a> LossEstimator<'a> {
    /// Create a new loss estimator.
    pub fn new(
        train_dataloader: &'a mut Loader,
        valid_dataloader: &'a mut Loader,
        loss: fn(&Tensor, &Tensor) -> Tensor,
    ) -> Self {
        Self {
            train_dataloader,
            valid_dataloader,
            loss,
        }
    }

    /// Estimate the loss of a model on the training and validation sets.
    pub fn estimate_loss(
        &mut self,
        model: &dyn LMModel,
        train_iters: usize,
        eval_iters: usize,
        progress_callback: &mut impl ProgressReporter,
    ) -> LossEstimates {
        let mut train_loss = 0.0;
        let mut valid_loss = 0.0;

        let mut n_train_batches = 0;
        let mut n_valid_batches = 0;

        progress_callback.train_loss_start(train_iters);

        while let Some((xs, ys)) = self.train_dataloader.next_batch() {
            let logits = model.forward_t(&xs, false);
            train_loss += f64::from((self.loss)(&logits, &ys));
            n_train_batches += 1;

            if n_train_batches % 10 == 0 {
                progress_callback.train_loss_progress(n_train_batches);
            }

            if n_train_batches >= train_iters {
                break;
            }
        }
        let train_loss = train_loss / n_train_batches as f64;

        progress_callback.train_loss_end(train_loss);

        progress_callback.valid_loss_start(eval_iters);

        while let Some((xs, ys)) = self.valid_dataloader.next_batch() {
            let logits = model.forward_t(&xs, false);
            valid_loss += f64::from((self.loss)(&logits, &ys));
            n_valid_batches += 1;

            if n_valid_batches % 10 == 0 {
                progress_callback.valid_loss_progress(n_valid_batches);
            }

            if n_valid_batches >= eval_iters {
                break;
            }
        }

        let valid_loss = valid_loss / n_valid_batches as f64;

        progress_callback.valid_loss_end(valid_loss);

        LossEstimates {
            train_loss,
            valid_loss,
        }
    }
}
