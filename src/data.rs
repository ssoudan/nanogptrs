use rand::prelude::*;

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
    data: Vec<i64>,
    vocab: Vocab,
}

impl TokenizedData {
    /// Create a new tokenized data from a string and a vocabulary
    pub fn new(data: &str, vocab: Vocab) -> Self {
        let data = vocab.encode(data);
        Self { data, vocab }
    }

    /// Return a reference to the vocabulary
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Return the number of tokens
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the data is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return a slice of the data
    pub fn slice(&self, start: usize, end: usize) -> &[i64] {
        &self.data[start..end]
    }
}

type Batch = (Vec<Vec<i64>>, Vec<Vec<i64>>);

/// Dataloader for tokenized data
/// Samples are tuples of the form (sample, target).
/// The sample is a tensor of size `batch_size x data_len`.
/// The target is a tensor of size `batch_size x data_len`.
///
/// Targets are the next token in the sequence.
///
/// Batches are made of `batch_size` random samples.
///
/// For example, if data is [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12], `data_len` is 3 and `batch_size` is 2,
/// the data loader could return the following batches:
/// - batch 1: (tensor([[0, 1, 2], [3, 4, 5]]), tensor([[1, 2, 3], [4, 5, 6]]))
/// - batch 2: (tensor([[6, 7, 8], [9, 10, 11]]), tensor([[7, 8, 9], [10, 11, 12]]))
///
pub struct Loader {
    data: TokenizedData,
    batch_size: usize,
    seq_len: usize,
    /// The number of (complete) batches
    n_batches: usize,
    /// The number of unique sequences of length `seq_len` in the data
    n_samples: usize,
    /// The order of the batches
    order: Option<Vec<usize>>,
    /// The current position in the order
    pos: usize,
}

impl Loader {
    /// Create a new data loader from a string and a vocabulary
    pub fn new(data: &str, seq_len: usize, batch_size: usize) -> Self {
        let tokenized_data = TokenizedData::new(data, Vocab::new(data));
        // The number of unique sequences of length `seq_len+1` in the data
        let n_samples = tokenized_data.len() - seq_len;
        // The number of (complete) batches
        let n_batches = n_samples / batch_size;

        Self {
            data: tokenized_data,
            batch_size,
            n_samples,
            n_batches,
            seq_len,
            order: None,
            pos: 0,
        }
    }

    /// Create a new data loader from tokenized data
    pub fn from_tokenized_data(data: TokenizedData, seq_len: usize, batch_size: usize) -> Self {
        // The number of unique sequences of length `seq_len+1` in the data
        let n_samples = data.len() - seq_len;
        // The number of (complete) batches
        let n_batches = n_samples / batch_size;

        Self {
            data,
            batch_size,
            n_samples,
            n_batches,
            seq_len,
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
                let sample = self.data.slice(pos , pos + self.seq_len);
                let target = self.data.slice(pos + 1, pos + self.seq_len + 1);
                samples.push(sample.to_vec());
                targets.push(target.to_vec());
            }
        } else {
            for i in 0..self.batch_size {
                let pos = self.pos + i;
                let sample = self.data.slice(pos , pos + self.seq_len);
                let target = self.data.slice(pos + 1 , pos + self.seq_len + 1 );
                samples.push(sample.to_vec());
                targets.push(target.to_vec());
            }
        }

        self.pos += self.batch_size;

        Some((samples, targets))
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
        let seq_len = 3;
        let batch_size = 2;
        let mut loader = Loader::new(data, seq_len, batch_size);

        assert_eq!(loader.n_samples(), 4);
        assert_eq!(loader.n_batches(), 2);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples, vec![vec![0, 1, 2], vec![1, 2, 3]]);
        assert_eq!(targets, vec![vec![1, 2, 3], vec![2, 3, 4]]);
        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples, vec![vec![2, 3, 4], vec![3, 4, 5]]);
        assert_eq!(targets, vec![vec![3, 4, 5], vec![4, 5, 6]]);
    }

    #[test]
    fn test_data_loader_shorter() {
        // [0, 1, 2, 3, 4, 5]
        // [a, b, c, d, e, f]

        // batches of size 2, sequence length 3

        // batch 1: (tensor([[0, 1, 2], [1, 2, 3]]), tensor([[1, 2, 3], [2, 3, 4]]))
        // with a leftover sample: (tensor([[2, 3, 4]]), tensor([[3, 4, 5]]))

        let data = "abcdef";
        let seq_len = 3;
        let batch_size = 2;
        let mut loader = Loader::new(data, seq_len, batch_size);

        assert_eq!(loader.n_samples(), 3);
        assert_eq!(loader.n_batches(), 1);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples, vec![vec![0, 1, 2], vec![1, 2, 3]]);
        assert_eq!(targets, vec![vec![1, 2, 3], vec![2, 3, 4]]);
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
        let seq_len = 3;
        let batch_size = 2;
        let mut loader = Loader::new(data, seq_len, batch_size);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);

        loader.shuffle(&mut rng);

        assert_eq!(loader.n_samples(), 4);
        assert_eq!(loader.n_batches(), 2);

        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples, vec![vec![1, 2, 3], vec![0, 1, 2],]);
        assert_eq!(targets, vec![vec![2, 3, 4], vec![1, 2, 3]]);
        let (samples, targets) = loader.next_batch().unwrap();
        assert_eq!(samples, vec![vec![2, 3, 4], vec![3, 4, 5]]);
        assert_eq!(targets, vec![vec![3, 4, 5], vec![4, 5, 6]]);
        assert!(loader.next_batch().is_none());
    }
}
