//! NanoGPTRS: A rust implementation of the NanoGPT
use nanogptrs::data::{load_file, Loader, TokenizedData, Vocab};
use rand_chacha::rand_core::SeedableRng;
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
    let valid_data = TokenizedData::new(valid_data, vocab);

    let batch_size = 32;
    let seq_len = 32;

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
}
