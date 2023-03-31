use rand::prelude::*;
use tch::{Device, Tensor};

/// Load data/input.txt and return a string
pub fn load_file() -> String {
    let path = "data/input.txt";
    std::fs::read_to_string(path).expect("Unable to read file")
}

/// Vocabulary
#[derive(Clone, Debug)]
pub struct Vocab {
    chars: Vec<char>,
}

impl Vocab {
    /// Create a new vocabulary from a string
    pub fn new(data: &str) -> Self {
        let mut chars: Vec<char> = data.chars().collect();
        chars.sort();
        chars.dedup();
        Self { chars }
    }

    /// Encode a character
    fn encode_char(&self, c: char) -> i64 {
        self.chars.iter().position(|&x| x == c).unwrap() as i64
    }

    /// Decode a character
    fn decode_char(&self, i: i64) -> char {
        self.chars[i as usize]
    }

    /// Encode a string
    pub fn encode(&self, s: &str) -> Vec<i64> {
        s.chars().map(|c| self.encode_char(c)).collect()
    }

    /// Decode a string
    pub fn decode(&self, v: &[i64]) -> String {
        v.iter().map(|&i| self.decode_char(i)).collect()
    }

    /// Return the size of the vocabulary
    pub fn size(&self) -> usize {
        self.chars.len()
    }

    /// Return a reference to the characters
    pub fn chars(&self) -> &Vec<char> {
        &self.chars
    }
}

/// Tokenized data
pub struct TokenizedData {
    data: Tensor,
    vocab: Vocab,
}

impl Clone for TokenizedData {
    fn clone(&self) -> Self {
        Self {
            data: self.data.copy(),
            vocab: self.vocab.clone(),
        }
    }
}

impl TokenizedData {
    /// Create a new tokenized data from a string and a vocabulary
    pub fn new(data: &str, vocab: Vocab, device: Device, kind: tch::Kind) -> Self {
        let data = vocab.encode(data);
        let data = Tensor::of_slice(&data).to_device(device).to_kind(kind);

        Self { data, vocab }
    }

    /// Return a reference to the vocabulary
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Return the number of tokens
    pub fn len(&self) -> usize {
        self.data.size1().unwrap() as usize
    }

    /// True if the data is empty
    pub fn is_empty(&self) -> bool {
        self.data.size1().unwrap() == 0
    }

    /// Return a slice of the data
    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        self.data.slice(0, start as i64, end as i64, 1)
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
/// For example, if data is [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12], `block_size` is 3 and `batch_size` is 2,
/// the data loader could return the following batches:
/// - batch 1: (tensor([[0, 1, 2], [3, 4, 5]]), tensor([[1, 2, 3], [4, 5, 6]]))
/// - batch 2: (tensor([[6, 7, 8], [9, 10, 11]]), tensor([[7, 8, 9], [10, 11, 12]]))
///
#[derive(Clone)]
pub struct Loader {
    data: TokenizedData,
    batch_size: usize,
    block_size: usize,
    /// The number of (complete) batches
    n_batches: usize,
    /// The number of unique sequences of length `block_size` in the data
    n_samples: usize,
    /// The order of the batches
    order: Option<Vec<usize>>,
    /// The current position in the order
    pos: usize,
}

impl Loader {
    /// Create a new data loader from a string and a vocabulary
    pub fn new(
        data: &str,
        block_size: usize,
        batch_size: usize,
        device: Device,
        kind: tch::Kind,
    ) -> Self {
        let tokenized_data = TokenizedData::new(data, Vocab::new(data), device, kind);
        // The number of unique sequences of length `block_size+1` in the data
        let n_samples = tokenized_data.len() - block_size;
        // The number of (complete) batches
        let n_batches = n_samples / batch_size;

        Self {
            data: tokenized_data,
            batch_size,
            n_samples,
            n_batches,
            block_size,
            order: None,
            pos: 0,
        }
    }

    /// Create a new data loader from tokenized data
    pub fn from_tokenized_data(data: TokenizedData, block_size: usize, batch_size: usize) -> Self {
        // The number of unique sequences of length `block_size+1` in the data
        let n_samples = data.len() - block_size;
        // The number of (complete) batches
        let n_batches = n_samples / batch_size;

        Self {
            data,
            batch_size,
            n_samples,
            n_batches,
            block_size,
            order: None,
            pos: 0,
        }
    }

    /// Return a reference to the vocabulary
    pub fn vocab(&self) -> &Vocab {
        self.data.vocab()
    }

    /// Return the number of batches
    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    /// Return the number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Pick a random order for the sequences to be picked to make the batches
    /// Also reset the position in the order. This means that `n_batches()` batches are available
    /// from here.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        let mut order: Vec<usize> = (0..self.n_samples).collect();
        order.shuffle(rng);
        self.order = Some(order);
        self.pos = 0;
    }

    /// Returns the next batch
    /// If `shuffle` has been called, the order of the batches is random.
    /// Otherwise, the batches are returned in the order of the data.
    pub fn next_batch(&mut self) -> Option<Batch> {
        let mut samples = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        // ensure there is enough data left to make a batch
        if self.pos + self.batch_size > self.n_samples {
            return None;
        }

        if let Some(order) = &self.order {
            for i in 0..self.batch_size {
                let pos = order[self.pos + i];
                let sample = self.data.slice(pos, pos + self.block_size);
                let target = self.data.slice(pos + 1, pos + self.block_size + 1);
                samples.push(sample);
                targets.push(target);
            }
        } else {
            for i in 0..self.batch_size {
                let pos = self.pos + i;
                let sample = self.data.slice(pos, pos + self.block_size);
                let target = self.data.slice(pos + 1, pos + self.block_size + 1);
                samples.push(sample);
                targets.push(target);
            }
        }

        self.pos += self.batch_size;

        let samples = Tensor::stack(&samples, 0);
        let targets = Tensor::stack(&targets, 0);

        Some((samples, targets))
    }
}

/// Test the data loader
#[cfg(test)]
mod tests {
    use super::*;

    fn to_tensor(batches: Vec<Vec<i64>>) -> Tensor {
        let batch_size = batches.len();

        let data: Vec<i64> = batches.into_iter().flatten().collect();
        Tensor::of_slice(&data).reshape(&[batch_size as i64, -1])
    }

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
        let mut loader = Loader::new(data, block_size, batch_size, Device::Cpu, tch::Kind::Int64);

        assert_eq!(loader.n_samples(), 4);
        assert_eq!(loader.n_batches(), 2);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int64);
        assert_eq!(samples, to_tensor(vec![vec![0, 1, 2], vec![1, 2, 3]]));
        assert_eq!(targets, to_tensor(vec![vec![1, 2, 3], vec![2, 3, 4]]));
        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int64);
        assert_eq!(samples, to_tensor(vec![vec![2, 3, 4], vec![3, 4, 5]]));
        assert_eq!(targets, to_tensor(vec![vec![3, 4, 5], vec![4, 5, 6]]));
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
        let mut loader = Loader::new(data, block_size, batch_size, Device::Cpu, tch::Kind::Int);

        assert_eq!(loader.n_samples(), 3);
        assert_eq!(loader.n_batches(), 1);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int);
        assert_eq!(samples, to_tensor(vec![vec![0, 1, 2], vec![1, 2, 3]]));
        assert_eq!(targets, to_tensor(vec![vec![1, 2, 3], vec![2, 3, 4]]));
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
        let mut loader = Loader::new(data, block_size, batch_size, Device::Cpu, tch::Kind::Int);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);

        loader.shuffle(&mut rng);

        assert_eq!(loader.n_samples(), 4);
        assert_eq!(loader.n_batches(), 2);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int);
        assert_eq!(samples, to_tensor(vec![vec![1, 2, 3], vec![0, 1, 2]]));
        assert_eq!(targets, to_tensor(vec![vec![2, 3, 4], vec![1, 2, 3]]));
        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples.kind(), tch::Kind::Int);
        assert_eq!(samples, to_tensor(vec![vec![2, 3, 4], vec![3, 4, 5]]));
        assert_eq!(targets, to_tensor(vec![vec![3, 4, 5], vec![4, 5, 6]]));
        assert!(loader.next_batch().is_none());
    }
}
