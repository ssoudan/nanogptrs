use crate::data::Loader;
use tch::nn::ModuleT;
use tch::{Device, Tensor};

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
    /// Called to update the progress of the loss estimation for the training sets.
    fn train_loss_progress(&mut self, current_train_batches: usize);
    /// Called at the end of the loss estimation for the training sets.
    fn train_loss_end(&mut self, train_loss: f64);
    /// Called at the start of the loss estimation for the validation sets.
    fn valid_loss_start(&mut self, total_valid_batches: usize);
    /// Called to update the progress of the loss estimation for the validation sets.
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

/// Estimate the loss of a model on the training and validation sets.
pub fn estimate_loss(
    train_dataloader: &mut Loader,
    valid_dataloader: &mut Loader,
    model: &impl ModuleT,
    device: Device,
    train_iters: usize,
    eval_iters: usize,
    progress_callback: &mut impl ProgressReporter,
    loss: fn(&Tensor, &Tensor) -> Tensor,
) -> LossEstimates {
    let mut train_loss = 0.0;
    let mut valid_loss = 0.0;

    let mut n_train_batches = 0;
    let mut n_valid_batches = 0;

    progress_callback.train_loss_start(n_train_batches);

    while let Some((xs, ys)) = train_dataloader.next_batch() {
        let xs = xs.to(device);
        let ys = ys.to(device);
        let logits = model.forward_t(&xs, false);
        train_loss += f64::from(loss(&logits, &ys));
        n_train_batches += 1;

        if n_train_batches % 100 == 0 {
            progress_callback.train_loss_progress(n_train_batches);
        }

        if n_train_batches >= train_iters {
            break;
        }
    }
    let train_loss = train_loss / n_train_batches as f64;

    progress_callback.train_loss_end(train_loss);

    progress_callback.valid_loss_start(n_valid_batches);

    while let Some((xs, ys)) = valid_dataloader.next_batch() {
        let xs = xs.to(device);
        let ys = ys.to(device);
        let logits = model.forward_t(&xs, false);
        valid_loss += f64::from(loss(&logits, &ys));
        n_valid_batches += 1;

        if n_valid_batches % 100 == 0 {
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

// TODO(ssoudan) test
