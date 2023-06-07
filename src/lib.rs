//! # The `nanogptrs` crate

/// The `data` module contains the structs and functions for loading, tokenizing
/// the data and generating batches.
pub mod data;

/// The `learn` module contains the structs and functions for training and
/// evaluating the model.
pub mod learn;

/// The `estimate` module contains the structs and functions for estimating the
/// loss.
pub mod estimate;

/// The `model` module contains the structs and functions for the model.
pub mod model;

/// Configuration structs and functions.
pub mod config;

/// Actions
pub mod actions;
