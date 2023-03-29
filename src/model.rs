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

    /// Generate a sequence of tokens from a starting sequence of tokens
    /// and a maximum length.
    /// xs: the starting sequence of tokens of shape [batch_size, seq_len]
    /// max_len: the maximum length of the generated sequence
    /// return: the generated sequence of tokens of shape [batch_size, max_len]
    pub fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
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

    let loss = |logits: &Tensor, targets: &Tensor| {
        let (b, t, _) = logits.size3().unwrap();
        let logits = logits.view([b * t, vocab_size]);
        let targets = targets.view([b * t]);
        logits.cross_entropy_loss::<Tensor>(&targets, None, tch::Reduction::Mean, -100, 0.)
    };

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
