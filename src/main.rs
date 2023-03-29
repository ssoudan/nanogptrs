//! NanoGPTRS: A rust implementation of the NanoGPT
use nanogptrs::data::{load_file, Loader, TokenizedData, Vocab};
use nanogptrs::model::loss;
use rand_chacha::rand_core::SeedableRng;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::Tensor;

fn main() {
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

    let vs = tch::nn::VarStore::new(tch::Device::Cpu);

    let model = nanogptrs::model::BigramLanguageModel::new(&vs.root(), vocab.size() as i64);

    let xs = Tensor::zeros(&[1, 1], (tch::Kind::Int64, tch::Device::Cpu));
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

    let device = tch::Device::Cpu;

    for epoch in 0..n_epochs {
        println!("Epoch {}/{}", epoch + 1, n_epochs);
        let mut train_loss = 0.0;
        let mut train_n = 0;

        let train_start = std::time::Instant::now();

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
        }
        let train_time = train_start.elapsed().as_secs_f64();
        println!(
            "train_loss: {:.4} train_ppl: {:.4} train_time: {:.2}s",
            train_loss / train_n as f64,
            (train_loss / train_n as f64).exp(),
            train_time
        );

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
        }
        let valid_time = valid_start.elapsed().as_secs_f64();
        println!(
            "valid_loss: {:.4} valid_ppl: {:.4} valid_time: {:.2}s",
            valid_loss / valid_n as f64,
            (valid_loss / valid_n as f64).exp(),
            valid_time
        );
    }

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
