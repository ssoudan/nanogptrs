//! NanoGPTRS: A rust implementation of the NanoGPT

use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use nanogptrs::actions;
use nanogptrs::config::{Device, Model, TrainingParameters};

// FUTURE(ssoudan): weight decay
// FUTURE(ssoudan): lr scheduler

/// The action to perform.
#[derive(Subcommand, Debug, Clone)]
enum Action {
    /// Next tokens distribution
    NextToken {
        /// The model to use
        #[command(subcommand)]
        model: Model,

        /// The prompt to use
        #[arg(long, default_value = "Once upon a time ")]
        prompt: String,
    },
    /// Generate text
    Generate {
        /// The model to use
        #[command(subcommand)]
        model: Model,

        /// The number of tokens to generate
        #[arg(long, default_value_t = 128)]
        max_len: usize,

        /// The prompt to use
        #[arg(long, default_value = "Once upon a time ")]
        prompt: String,

        /// Seed for the random number generator
        #[arg(long, default_value = "42")]
        seed: i64,
    },
    /// Train the model
    Train {
        /// The model to use
        #[command(subcommand)]
        model: Model,

        /// Training parameters
        #[clap(flatten)]
        training_params: TrainingParameters,
    },
}

impl Default for Action {
    fn default() -> Self {
        Self::Generate {
            model: Model::default(),
            max_len: 128,
            prompt: "Once upon a time ".to_string(),
            seed: 1337,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The device to use
    #[arg(short, long, default_value = "cuda")]
    device: Device,

    /// Checkpoint to restore from
    #[arg(short, long)]
    restore_from: Option<String>,
    /// The action to perform
    #[command(subcommand)]
    action: Action,
}

fn main() {
    let args = Args::parse();

    let device = match args.device {
        Device::Cpu => tch::Device::Cpu,
        Device::Cuda => tch::Device::cuda_if_available(),
        #[cfg(target_arch = "aarch64")]
        Device::Mps => tch::Device::Mps,
    };

    // print a banner with nanoGPTrs
    println!(r#"nanoGPTrs"#);

    // if not built in release mode, print a big warning
    #[cfg(debug_assertions)]
    {
        println!("WARNING: This is a debug build. Things will be painfully slow.");
    }

    ///////
    println!("device: {:?}", device);

    let mut vs = tch::nn::VarStore::new(device);

    match args.action {
        Action::NextToken { model, prompt } => {
            // Build the model
            let (model, tokenizer) = actions::create_model(&vs, model);
            println!("[+] Got a model and a tokenizer");

            // Restore the model from the checkpoint
            if args.restore_from.is_some() {
                println!("[.] Restoring the model from the checkpoint");
                // Restore the model from the checkpoint
                vs.load(args.restore_from.as_ref().unwrap()).unwrap();
                println!("[+] Restored the model from the checkpoint")
            }

            // freeze
            vs.freeze();

            println!("[.] Next token probabilities for: [{}]...", prompt);
            let gen = actions::next_token(device, model, tokenizer.as_ref(), prompt);
            println!("[+] distribution: {:#?}", gen);
        }
        Action::Generate {
            model,
            max_len,
            prompt,
            seed,
        } => {
            // setting the seed
            tch::manual_seed(seed);

            // Build the model
            let (model, tokenizer) = actions::create_model(&vs, model);
            println!("[+] Got a model and a tokenizer");

            // Restore the model from the checkpoint
            if args.restore_from.is_some() {
                println!("[.] Restoring the model from the checkpoint");
                // Restore the model from the checkpoint
                vs.load(args.restore_from.as_ref().unwrap()).unwrap();
                println!("[+] Restored the model from the checkpoint")
            }

            // freeze
            vs.freeze();

            println!("[.] Generating text: [{}]...", prompt);
            let gen = actions::generate(device, model, tokenizer.as_ref(), prompt, max_len);
            println!("[+] Generated text: {}", gen);
        }
        Action::Train {
            model,
            training_params,
        } => {
            let model_name = match &model {
                Model::NanoGpt { .. } => "nanoGPT".to_string(),
                Model::Bigram { .. } => "bigram".to_string(),
                Model::GPT2 { args } => args.size.to_string(),
            };

            // Build the model
            let (model, tokenizer) = actions::create_model(&vs, model);
            println!("[+] Got a model and a tokenizer");

            // Restore the model from the checkpoint
            if args.restore_from.is_some() {
                println!("[.] Restoring the model from the checkpoint");
                // Restore the model from the checkpoint
                vs.load(args.restore_from.as_ref().unwrap()).unwrap();
                println!("[+] Restored the model from the checkpoint")
            }

            let run_name = training_params.run_name.clone().unwrap_or_else(|| {
                let now: DateTime<Utc> = Utc::now();
                format!("{}_{}", model_name, now.format("%Y%m%d_%H%M%S"))
            });

            // only allow [a-zA-Z0-9_-] in the run name
            let run_name = run_name
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect::<String>();

            println!("[.] Training the model - [{}]...", run_name);
            actions::train(vs, device, run_name, model, tokenizer, training_params);
            println!("[+] Trained the model");
        }
    }
}
