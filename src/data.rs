use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::BufRead;

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;
use tch::{Device, Tensor};

/// Load data/input.txt and return a string
pub fn load_file(path: &str) -> String {
    std::fs::read_to_string(path).expect(format!("Unable to read file {}", path).as_str())
}

/// Vocabulary
#[derive(Clone, Debug)]
pub struct Vocab {
    chars: Vec<char>,
}

// FUTURE(ssoudan) template the String part
/// A tokenizer
pub trait Tokenizer: Sync {
    /// Encode a string
    fn encode(&self, s: &str) -> Vec<i64>;

    /// Decode a string
    fn decode(&self, v: &[i64]) -> String;
}

impl Vocab {
    /// Create a new vocabulary from a string
    pub fn new(data: &str) -> Self {
        let mut chars: Vec<char> = data.chars().collect();
        chars.sort();
        chars.dedup();
        Self { chars }
    }

    /// Create a new vocabulary from a file
    pub fn from_file(path: &str) -> Self {
        let f = File::open(path).expect("Unable to open file");

        let chars: HashSet<char> = std::io::BufReader::new(f)
            .lines()
            .map(|l| l.expect("Unable to read line") + "\n")
            .map(|l| l.chars().sorted().dedup())
            .fold(HashSet::new(), |mut acc, x| {
                acc.extend(x);
                acc
            });

        let chars: String = chars.into_iter().sorted().collect();

        println!("chars: {:?}", chars);

        Self::new(&chars)
    }

    /// Encode a character
    fn encode_char(&self, c: char) -> i64 {
        let pos = self.chars.iter().position(|&x| x == c);

        match pos {
            Some(i) => i as i64,
            None => panic!(
                "Character [{}] not found in the vocabulary: {}",
                c,
                self.chars.iter().collect::<String>()
            ),
        }
    }

    /// Decode a character
    fn decode_char(&self, i: i64) -> char {
        self.chars[i as usize]
    }

    /// Return the size of the vocabulary
    pub fn size(&self) -> usize {
        self.chars.len()
    }
}

impl Tokenizer for Vocab {
    fn encode(&self, s: &str) -> Vec<i64> {
        // FUTURE(ssoudan) return a Result
        s.chars().map(|c| self.encode_char(c)).collect()
    }

    fn decode(&self, v: &[i64]) -> String {
        v.iter().map(|&i| self.decode_char(i)).collect()
    }
}

/// Tokenized data
struct TokenizedData {
    data: Tensor,
    device: Device,
    order: Option<VecDeque<usize>>,
    block_size: usize,
}

impl Clone for TokenizedData {
    fn clone(&self) -> Self {
        Self {
            data: self.data.copy(),
            device: self.device,
            block_size: self.block_size,
            order: None,
        }
    }
}

impl TokenizedData {
    /// Create a new tokenized data from a string and a vocabulary
    fn new(
        data: &str,
        tokenizer: &dyn Tokenizer,
        device: Device,
        kind: tch::Kind,
        block_size: usize,
    ) -> Self {
        let data = tokenizer.encode(data);

        // will get moved to the GPU later
        let data = Tensor::from_slice(&data)
            .to_device(Device::Cpu)
            .to_kind(kind);

        Self {
            data,
            device,
            order: None,
            block_size,
        }
    }

    /// Return the number of samples
    fn sample_count(&self) -> usize {
        self.data.size1().unwrap() as usize - self.block_size
    }

    /// Return a slice of the data
    fn next_sample(&mut self) -> Option<(Tensor, Tensor)> {
        let pos = self.order.as_mut()?.pop_front()?;

        let start = pos as i64;
        let end = (pos + self.block_size) as i64;

        let sample = self.data.slice(0, start, end, 1);
        let start = start + 1;
        let end = end + 1;

        let target = self.data.slice(0, start, end, 1);

        Some((sample, target))
    }

    fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        let mut order: Vec<usize> = (0..self.sample_count()).collect();
        order.shuffle(rng);
        let order = order.into_iter().collect();
        self.order = Some(order);
    }

    fn load(&mut self) {
        self.data = self.data.to_device(self.device);
    }

    fn unload(&mut self) {
        self.data = self.data.to(Device::Cpu);
    }
}

/// A batch of data
/// The first element is the input and the second element is the target.
/// The input is a tensor of size `batch_size x block_size`.
/// The target is a tensor of size `batch_size x block_size`.
type Batch = (Tensor, Tensor);

/// Dataloader for tokenized data
/// Samples are tuples of the form (sample, target).
/// The sample is a tensor of size `batch_size x block_size`.
/// The target is a tensor of size `batch_size x block_size`.
///
/// Targets are the next token in the sequence.
///
/// Batches are made of `batch_size` random samples.
///
/// For example, if data is [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
/// `block_size` is 3 and `batch_size` is 2, the data loader could return the
/// following batches:
/// - batch 1: (tensor([[0, 1, 2], [3, 4, 5]]), tensor([[1, 2, 3], [4, 5, 6]]))
/// - batch 2: (tensor([[6, 7, 8], [9, 10, 11]]), tensor([[7, 8, 9], [10, 11,
///   12]]))
#[derive(Clone)]
pub struct Loader {
    chunks: Vec<TokenizedData>,

    batch_size: usize,
    block_size: usize,
    /// The number of (complete) batches
    n_batches: usize,
    /// The number of unique sequences of length `block_size` in the data
    n_samples: usize,

    // Dequeue of the indices of chunks
    remaining_chunks: VecDeque<usize>,
}

impl Loader {
    /// Create a new data loader from a string and a vocabulary
    pub fn new(
        data: &str,
        tokenizer: &dyn Tokenizer,
        block_size: usize,
        batch_size: usize,
        device: Device,
        kind: tch::Kind,
    ) -> Self {
        // split data into blocks of size 16M chars
        const CHUNK_SIZE: usize = 2 << 24; // 16M chars

        // TODO(ssoudan) observer

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("TOKENIZING [{elapsed_precise}] {spinner:.green} [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Tokenizing data");
        pb.set_length(data.len() as u64 / CHUNK_SIZE as u64);
        let chunks = &data
            .chars()
            .chunks(CHUNK_SIZE)
            .into_iter()
            .map(|chunk| chunk.collect::<String>())
            .collect::<Vec<_>>();

        let chunks: Vec<_> = chunks
            .par_iter()
            .map(|chunk| {
                let t = TokenizedData::new(chunk, tokenizer, device, kind, block_size);
                pb.inc(1);
                t
            })
            .collect();
        // for chunk in &data.chars().par_chunks(CHUNK_SIZE) {
        //     let chunk = String::from_iter(chunk);
        //     chunks.push(TokenizedData::new(
        //         &chunk, tokenizer, device, kind, block_size,
        //     ));
        //     pb.inc(1);
        // }
        pb.finish();

        let chunk_samples: Vec<_> = chunks.iter().map(|d| d.sample_count()).collect();

        // The number of unique sequences of length `block_size+1` in the data
        let n_samples = chunk_samples.iter().sum();

        // The number of (complete) batches
        let n_batches = n_samples / batch_size;

        Self {
            chunks,
            batch_size,
            n_samples,
            n_batches,
            block_size,
            remaining_chunks: VecDeque::new(),
        }
    }

    /// Return the number of batches
    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    /// Return the number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Return the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Pick a random order for the sequences to be picked to make the batches
    /// Also reset the position in the order. This means that `n_batches()`
    /// batches are available from here.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        // unload the current chunk if any
        if let Some(current_chunk_idx) = self.remaining_chunks.front() {
            self.chunks[*current_chunk_idx].unload();
        }

        // shuffle the chunk order
        let mut chunks: Vec<usize> = (0..self.chunks.len()).collect();

        chunks.shuffle(rng);

        self.remaining_chunks = VecDeque::from(chunks);

        // shuffle the chunks
        for chunk in &mut self.chunks {
            chunk.shuffle(rng);
        }

        // load the first chunk
        self.chunks[self.remaining_chunks[0]].load();
    }

    fn next_sample(&mut self) -> Option<(Tensor, Tensor)> {
        while let Some(chunk_idx) = self.remaining_chunks.front() {
            let chunk = &mut self.chunks[*chunk_idx];

            let sample = chunk.next_sample();
            if sample.is_some() {
                return sample;
            }

            // We are done with this chunk
            // unload it
            chunk.unload();
            self.remaining_chunks.pop_front();

            // load the next chunk if any
            if let Some(next_chunk_idx) = self.remaining_chunks.front() {
                self.chunks[*next_chunk_idx].load();
            }
        }

        // We are done with all the chunks
        None
    }

    /// Returns the next batch
    /// If `shuffle` has been called, the order of the batches is random.
    /// Otherwise, the batches are returned in the order of the data.
    pub fn next_batch(&mut self) -> Option<Batch> {
        let mut samples = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let (sample, target) = self.next_sample()?;
            samples.push(sample);
            targets.push(target);
        }

        let samples = Tensor::stack(&samples, 0);
        let targets = Tensor::stack(&targets, 0);

        Some((samples, targets))
    }
}

/// GPT2 tokenizer
#[derive(Clone)]
pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Gpt2Tokenizer {
    /// Create a new tokenizer from a file
    pub fn new(path: String) -> Self {
        let tokenizer: tokenizers::Tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();

        Self { tokenizer }
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, s: &str) -> Vec<i64> {
        let encoding = self.tokenizer.encode(s, false).unwrap();
        encoding.get_ids().iter().map(|&x| x as i64).collect()
    }

    fn decode(&self, v: &[i64]) -> String {
        let v: Vec<u32> = v.iter().map(|&x| x as u32).collect();
        self.tokenizer.decode(&v, false).unwrap()
    }
}

/// Test the data loader
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader() {
        // [0, 1, 2, 3, 4, 5, 6]
        // [a, b, c, d, e, f, g]

        // batches of size 2, sequence length 3

        // batch 1: (tensor([[0, 1, 2], [1, 2, 3]]), tensor([[1, 2, 3], [2, 3, 4]]))
        // batch 2: (tensor([[2, 3, 4], [3, 4, 5]]), tensor([[3, 4, 5], [4, 5, 6]]))

        let data = "abcdefg";
        let block_size = 3;
        let batch_size = 2;
        let tokenizer = Vocab::new(data);
        let tokenizer = Box::new(tokenizer);
        let mut loader = Loader::new(
            data,
            tokenizer.as_ref(),
            block_size,
            batch_size,
            Device::Cpu,
            tch::Kind::Int64,
        );

        assert_eq!(loader.n_samples(), 4);
        assert_eq!(loader.n_batches(), 2);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        loader.shuffle(&mut rng);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int64);
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&samples).unwrap(),
            vec![vec![0, 1, 2], vec![2, 3, 4]]
        );
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&targets).unwrap(),
            vec![vec![1, 2, 3], vec![3, 4, 5]]
        );
        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int64);
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&samples).unwrap(),
            vec![vec![1, 2, 3], vec![3, 4, 5]]
        );
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&targets).unwrap(),
            vec![vec![2, 3, 4], vec![4, 5, 6]]
        );
    }

    #[test]
    fn test_data_loader_shorter() {
        // [0, 1, 2, 3, 4, 5]
        // [a, b, c, d, e, f]

        // batches of size 2, sequence length 3

        // batch 1: (tensor([[0, 1, 2], [1, 2, 3]]), tensor([[1, 2, 3], [2, 3, 4]]))
        // with a leftover sample: (tensor([[2, 3, 4]]), tensor([[3, 4, 5]]))

        let data = "abcdef";
        let block_size = 3;
        let batch_size = 2;
        let tokenizer = Vocab::new(data);
        let tokenizer = Box::new(tokenizer);
        let mut loader = Loader::new(
            data,
            tokenizer.as_ref(),
            block_size,
            batch_size,
            Device::Cpu,
            tch::Kind::Int,
        );

        assert_eq!(loader.n_samples(), 3);
        assert_eq!(loader.n_batches(), 1);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        loader.shuffle(&mut rng);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int);
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&samples).unwrap(),
            vec![vec![2, 3, 4], vec![1, 2, 3]]
        );
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&targets).unwrap(),
            vec![vec![3, 4, 5], vec![2, 3, 4]]
        );
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_data_loader_shuffle() {
        // [0, 1, 2, 3, 4, 5, 6]
        // [a, b, c, d, e, f, g]

        // batches of size 2, sequence length 3

        // batch 1: (tensor([[1, 2, 3], [0, 1, 2]]), tensor([[2, 3, 4], [1, 2, 3]]))
        // batch 2: (tensor([[2, 3, 4], [3, 4, 5]]), tensor([[3, 4, 5], [4, 5, 6]]))

        let data = "abcdefg";
        let block_size = 3;
        let batch_size = 2;
        let tokenizer = Vocab::new(data);
        let tokenizer = Box::new(tokenizer);
        let mut loader = Loader::new(
            data,
            tokenizer.as_ref(),
            block_size,
            batch_size,
            Device::Cpu,
            tch::Kind::Int,
        );
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);

        loader.shuffle(&mut rng);

        assert_eq!(loader.n_samples(), 4);
        assert_eq!(loader.n_batches(), 2);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int);
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&samples).unwrap(),
            vec![vec![1, 2, 3], vec![0, 1, 2]]
        );
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&targets).unwrap(),
            vec![vec![2, 3, 4], vec![1, 2, 3]]
        );
        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int);
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&samples).unwrap(),
            vec![vec![2, 3, 4], vec![3, 4, 5]]
        );
        assert_eq!(
            <Vec::<Vec<i64>>>::try_from(&targets).unwrap(),
            vec![vec![3, 4, 5], vec![4, 5, 6]]
        );
        assert!(loader.next_batch().is_none());
    }
}
