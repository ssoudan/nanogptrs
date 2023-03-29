use tch::nn;
use tch::IndexOp;
use tch::Tensor;

/// BigramLanguageModel is a language model that uses the current token to predict the next token.
#[derive(Debug)]
pub struct BigramLanguageModel {
    /// The embedding layer
    embedding: nn::Embedding,
    // /// The linear layer
    // linear: nn::Linear,
}

impl BigramLanguageModel {
    /// Create a new BigramLanguageModel
    pub fn new(vs: &nn::Path, vocab_size: i64) -> Self {
        let embedding = nn::embedding(vs / "embedding", vocab_size, vocab_size, Default::default());
        // let linear = nn::linear(vs / "linear", embedding_dim, hidden_dim, Default::default());

        Self { embedding }
    }

    /// Forward pass
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.embedding)
        // xs.apply(&self.linear)
    }
}

impl LMModel for BigramLanguageModel {
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
        let mut xs = xs;
        for _ in 0..max_len {
            // take the last logits
            let logits = self.forward(&xs).i((.., -1, ..));
            //  apply softmax to get the probabilities of the next token
            let probs = logits.softmax(-1, tch::Kind::Float);
            // sample the next token
            let next_token = probs.multinomial(1, true);
            xs = Tensor::cat(&[xs, next_token], 1);
        }
        xs
    }
}

impl nn::ModuleT for BigramLanguageModel {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// NanoGpt is a language model.
#[derive(Debug)]
pub struct NanoGpt {
    /// The embedding layer
    token_embedding: nn::Embedding,
    /// The position embedding layer
    pub position_embedding: nn::Embedding,
    /// LM head
    lm_head: nn::Linear,
    // /// The linear layer
    // linear: nn::Linear,
    /// The embedding size
    n_embd: i64,
    /// The vocabulary size
    vocab_size: i64,
}

impl NanoGpt {
    /// Create a new NanoGpt
    pub fn new(vs: &nn::Path, vocab_size: i64, seq_len: i64, n_embd: i64) -> Self {
        let token_embedding =
            nn::embedding(vs / "embedding", vocab_size, n_embd, Default::default());

        let position_embedding = nn::embedding(
            vs / "position_embedding",
            seq_len,
            n_embd,
            Default::default(),
        );

        let lm_head = nn::linear(vs / "lm_head", n_embd, vocab_size, Default::default());

        Self {
            token_embedding,
            position_embedding,
            lm_head,
            n_embd,
            vocab_size,
        }
    }

    /// Forward pass
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let (b, t) = xs.size2().unwrap();

        let tok_emb = xs.apply(&self.token_embedding); // [batch_size, seq_len, n_embd]

        let device = xs.device();

        let pos = Tensor::arange(t, (tch::Kind::Int64, device)); // [seq_len=t]

        let pos = pos.unsqueeze(0); // [1, seq_len=t]
        assert_eq!(pos.size(), &[1, t]);

        let pos = pos.repeat(&[b, 1, 1]);
        assert_eq!(pos.size(), &[b, 1, t]);

        let pos_emb = pos.apply(&self.position_embedding); // [seq_len, n_embd]
        assert_eq!(pos_emb.size(), &[t, self.n_embd]);

        let x = tok_emb + pos_emb; // [batch_size, seq_len, n_embd]
        assert_eq!(x.size(), &[b, t, self.n_embd]);

        let x = x.apply(&self.lm_head); // [batch_size, seq_len, vocab_size]
        assert_eq!(x.size(), &[b, t, self.vocab_size]);
        x
    }
}

impl LMModel for NanoGpt {
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
        let mut xs = xs;
        for _ in 0..max_len {
            // take the last logits
            let logits = self.forward(&xs).i((.., -1, ..));
            //  apply softmax to get the probabilities of the next token
            let probs = logits.softmax(-1, tch::Kind::Float);
            // sample the next token
            let next_token = probs.multinomial(1, true);
            xs = Tensor::cat(&[xs, next_token], 1);
        }
        xs
    }
}

impl nn::ModuleT for NanoGpt {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// LMModel is a language model.
pub trait LMModel: nn::ModuleT {
    /// Generate a sequence of tokens from a starting sequence of tokens
    /// and a maximum length.
    ///
    /// xs: the starting sequence of tokens of shape [batch_size, seq_len]
    /// max_len: the maximum length of the generated sequence
    /// return: the generated sequence of tokens of shape [batch_size, max_len]
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor;
}

/// Compute the loss
///
/// Use cross entropy loss.
pub fn loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    let (b, t, c) = logits.size3().unwrap();

    logits
        .view([b * t, c])
        .cross_entropy_for_logits(&targets.view([b * t]))
}

/// Test the BigramLanguageModel
#[test]
fn test_bigram_language_model() {
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let vocab_size = 100;
    let batch_size = 2;
    let seq_len = 5;

    let model = BigramLanguageModel::new(&vs.root(), vocab_size);
    let xs = Tensor::of_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to_kind(tch::Kind::Int64);
    let xs = xs.view([batch_size, seq_len]);
    println!("xs: {:?}", xs);
    let (b, t) = xs.size2().unwrap();
    assert_eq!(b, batch_size);
    assert_eq!(t, seq_len);

    let logits = model.forward(&xs);
    assert_eq!(logits.size(), [batch_size, seq_len, vocab_size]);

    let loss = loss(&logits, &xs);
    println!("loss: {:?}", loss);
    // we expect the loss to be close to -ln(1/vocab_size) = 4.17

    // use 0 as start of sequence token - this is '\n' with our data and tokenizer.
    let xs = Tensor::zeros(&[batch_size, 1], (tch::Kind::Int64, tch::Device::Cpu));
    let max_len = 10;
    let ys = model.generate(xs, max_len);
    println!("generated: {:?}", ys);

    // decode the generated sequence of tokens
    let ys = ys.to_kind(tch::Kind::Int64);

    println!("generated: {:?}", ys);

    let first = ys.i((0, ..));
    let first: Vec<i64> = first.into();
    println!("first: {:?}", first);

    let second = ys.i((1, ..));
    let second: Vec<i64> = second.into();
    println!("second: {:?}", second);
}

// TODO(ssoudan): test the LanguageModel
