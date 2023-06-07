use rand::{Rng, SeedableRng};
use tch::nn::{OptimizerConfig, VarStore};
use tch::Tensor;

use crate::config::{GPT2Args, Gpt2Size, LearnConfig, Model, NanoGptArgs, TrainingParameters};
use crate::data::{load_file, Gpt2Tokenizer, Loader, TokenizedData, Tokenizer, Vocab};
use crate::estimate::LossEstimator;
use crate::learn::logger::TensorboardReporter;
use crate::learn::{Observer, PbProgressReporter, ProgressReporter};
use crate::model::nano::NanoGptConfig;
use crate::model::{loss, LanguageModel};

/// Next token probabilities
pub fn next_token(
    device: tch::Device,
    model: Box<dyn LanguageModel>,
    tokenizer: Box<dyn Tokenizer>,
    xs: String,
) -> Vec<(String, f64)> {
    // TODO(ssoudan) ability to insert hooks in the forward pass

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
pub fn generate(
    device: tch::Device,
    model: Box<dyn LanguageModel>,
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
pub fn train(
    mut vs: VarStore,
    device: tch::Device,
    run_name: String,
    model: Box<dyn LanguageModel>,
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
    let mut observer = Observer::default().with(Box::<PbProgressReporter>::default());
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

/// Create the model from the model parameters
pub fn create_model(
    vs: &mut VarStore,
    model: Model,
) -> (Box<dyn LanguageModel>, Box<dyn Tokenizer>) {
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
                Box::new(crate::model::nano::NanoGpt::new(&vs.root(), config)),
                Box::new(vocab),
            )
        }
        Model::Bigram { vocab_file } => {
            // Load the data
            let data = load_file(&vocab_file);
            // build the vocabulary
            let vocab = Vocab::new(&data);
            (
                Box::new(crate::model::bigram::BigramLanguageModel::new(
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
                Box::new(crate::model::nano::NanoGpt::new(&vs.root(), config)),
                Box::new(tokenizer),
            )
        }
    }
}

/// Train the model
#[allow(clippy::borrowed_box, clippy::too_many_arguments)]
pub fn learn<R: Rng>(
    learn_config: LearnConfig,
    run_name: String,
    train_dataloader: &mut Loader,
    valid_dataloader: &mut Loader,
    mut rng: &mut R,
    vs: &mut tch::nn::VarStore,
    model: &Box<dyn LanguageModel>,
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
