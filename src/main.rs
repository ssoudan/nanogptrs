//! NanoGPTRS: A rust implementation of the NanoGPT
use indicatif::{ProgressBar, ProgressStyle};
use nanogptrs::data::{load_file, Loader, TokenizedData, Vocab};
use nanogptrs::model::loss;
use rand_chacha::rand_core::SeedableRng;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::Tensor;

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

    let multi_bar = indicatif::MultiProgress::new();

    // epoch bar
    let epoch_bar = multi_bar.add(ProgressBar::new(n_epochs as u64));
    epoch_bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] Epoch {pos:>7}/{len:7} {msg.4}",
            )
            .unwrap()
            .progress_chars("##-"),
    );

    for _epoch in 0..n_epochs {
        epoch_bar.tick();
        let mut train_loss = 0.0;
        let mut train_n = 0;

        let train_start = std::time::Instant::now();

        let training_bar = multi_bar.add(ProgressBar::new(train_dataloader.n_batches() as u64));
        training_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:20.green/blue}] Training {human_pos:>7}/{human_len:7} @ {per_sec:.1} {msg.4}",
                )
                .unwrap()
                .progress_chars("##-"),
        );
        training_bar.tick();

        train_dataloader.shuffle(&mut rng);
        while let Some((xs, ys)) = train_dataloader.next_batch() {
            let xs: Vec<i64> = xs.into_iter().flatten().collect();
            let ys: Vec<i64> = ys.into_iter().flatten().collect();

            let xs = Tensor::of_slice(&xs)
                .to_kind(tch::Kind::Int64)
                .to(device)
                .view([batch_size as i64, seq_len as i64]);
            let ys = Tensor::of_slice(&ys)
                .to_kind(tch::Kind::Int64)
                .to(device)
                .view([batch_size as i64, seq_len as i64]);

            let logits = model.forward_t(&xs, true);

            let loss = loss(&logits, &ys);

            // opt.backward_step(&loss);
            opt.zero_grad();
            loss.backward();
            opt.step();

            let loss: f64 = loss.into();

            train_loss += loss;
            train_n += 1;

            if train_n % 1000 == 0 {
                training_bar.inc(1000);
            }
        }

        training_bar.finish_and_clear();

        let train_time = train_start.elapsed().as_secs_f64();
        epoch_bar.set_message(format!(
            "[train] loss: {:.4} ppl: {:.4} time: {:.2}s",
            train_loss / train_n as f64,
            (train_loss / train_n as f64).exp(),
            train_time
        ));

        // validation
        let validation_bar = multi_bar.add(ProgressBar::new(valid_dataloader.n_batches() as u64));
        validation_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:20.yellow/blue}] Validation {human_pos:>7}/{human_len:7} @ {per_sec:.1} {msg.4}",
                )
                .unwrap()
                .progress_chars("##-"),
        );

        validation_bar.tick();

        let mut valid_loss = 0.0;
        let mut valid_n = 0;
        let valid_start = std::time::Instant::now();
        valid_dataloader.shuffle(&mut rng);
        while let Some((xs, ys)) = valid_dataloader.next_batch() {
            let xs: Vec<i64> = xs.into_iter().flatten().collect();
            let ys: Vec<i64> = ys.into_iter().flatten().collect();

            let xs = Tensor::of_slice(&xs)
                .to_kind(tch::Kind::Int64)
                .to(device)
                .view([batch_size as i64, seq_len as i64]);
            let ys = Tensor::of_slice(&ys)
                .to_kind(tch::Kind::Int64)
                .to(device)
                .view([batch_size as i64, seq_len as i64]);

            let logits = model.forward_t(&xs, false);

            let loss = loss(&logits, &ys);

            // convert to a scalar
            let loss: f64 = loss.into();

            valid_loss += loss;
            valid_n += 1;

            if valid_n % 200 == 0 {
                validation_bar.inc(200);
            }
        }
        let valid_time = valid_start.elapsed().as_secs_f64();
        epoch_bar.set_message(format!(
            "[train] loss: {:.4} ppl: {:.4} time: {:.2}s [valid] loss: {:.4} ppl: {:.4} time: {:.2}s",
            train_loss / train_n as f64,
            (train_loss / train_n as f64).exp(),
            train_time,
            valid_loss / valid_n as f64,
            (valid_loss / valid_n as f64).exp(),
            valid_time
        ));

        validation_bar.finish_and_clear();

        epoch_bar.inc(1);
    }

    epoch_bar.finish();

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
