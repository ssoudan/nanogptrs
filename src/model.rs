use std::borrow::Borrow;
use tch::nn;
use tch::nn::{ModuleT, Path};
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

/// One head self-attention.
#[derive(Debug)]
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    mask: Tensor,
    head_size: i64,
}

impl Head {
    /// Create a new Head.
    ///
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `seq_len` - The sequence length.
    /// * `n_emb` - The embedding size.
    /// * `head_size` - The size of the head.
    ///
    /// # Returns
    /// A new Head.
    ///
    /// # Notes
    /// The input of `forward` is expected to be of shape `[batch_size, seq_len, C]`.
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, seq_len: i64, n_emb: i64, head_size: i64) -> Self {
        let vs = vs.borrow();

        let device = vs.device();

        let key = nn::linear(vs / "key", n_emb, head_size, Default::default());
        let query = nn::linear(vs / "query", n_emb, head_size, Default::default());
        let value = nn::linear(vs / "value", n_emb, head_size, Default::default());
        let mask = Tensor::ones(&[seq_len, seq_len], (tch::Kind::Float, device)).tril(0);
        Self {
            key,
            query,
            value,
            mask,
            head_size,
        }
    }
}

impl nn::ModuleT for Head {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let (b, t, _c) = xs.size3().unwrap();
        let head_size = self.head_size;

        // the dimension of the keys
        let d_k = head_size;

        let k = xs.apply(&self.key); // [b, t, head_size]
        assert_eq!(k.size(), &[b, t, head_size]);
        let q = xs.apply(&self.query); // [b, t, head_size]
        assert_eq!(q.size(), &[b, t, head_size]);
        let v = xs.apply(&self.value); // [b, t, head_size]
        assert_eq!(v.size(), &[b, t, head_size]);

        let wei = q.matmul(&k.transpose(-2, -1)) / (d_k as f64).sqrt();
        let wei = wei.masked_fill(&self.mask.eq(0.), f64::NEG_INFINITY);
        assert_eq!(wei.size(), &[b, t, t]);

        let wei = wei.softmax(-1, tch::Kind::Float);
        assert_eq!(wei.size(), &[b, t, t]);

        // weighted aggregation of the values
        let out = wei.matmul(&v); // [b, t, head_size]
        assert_eq!(out.size(), &[b, t, head_size]);

        out
    }
}

/// MultiHeadSelfAttention is a multi-head self-attention layer.
#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    heads: Vec<Head>,
}

impl MultiHeadSelfAttention {
    /// Create a new MultiHeadSelfAttention.
    /// # Arguments
    /// * `vs` - The path to the module.
    /// * `seq_len` - The sequence length.
    /// * `n_emb` - The embedding size.
    /// * `head_size` - The size of the head.
    /// * `n_head` - The number of heads.
    /// # Returns
    /// A new MultiHeadSelfAttention.
    pub fn new<'a, T: Borrow<Path<'a>>>(
        vs: T,
        seq_len: i64,
        n_emb: i64,
        head_size: i64,
        n_head: i64,
    ) -> Self {
        let vs = vs.borrow();

        let heads = (0..n_head)
            .map(|i| Head::new(vs / i, seq_len, n_emb, head_size))
            .collect();

        Self { heads }
    }
}

impl nn::ModuleT for MultiHeadSelfAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // concatenate the heads along the last dimension
        let heads = self
            .heads
            .iter()
            .map(|h| h.forward_t(xs, train))
            .collect::<Vec<_>>();

        Tensor::cat(&heads, -1)
    }
}

/// Feed forward layer
#[derive(Debug)]
struct FeedForward {
    net: nn::Linear,
}

impl FeedForward {
    /// Create a new FeedForward
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, n_embd: i64) -> Self {
        let vs = vs.borrow();
        let net = nn::linear(vs / "net", n_embd, n_embd, Default::default());
        Self { net }
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
    /// SA heads
    sa_heads: MultiHeadSelfAttention,
    /// Feed forward layer
    ffwd: FeedForward,
    /// The embedding size
    n_embd: i64,
    /// The vocabulary size
    vocab_size: i64,
    /// The sequence length
    seq_len: i64,
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

        // let sa_heads = Head::new(vs / "sa_head", seq_len, n_embd);
        assert!(n_embd % 4 == 0, "n_embd must be a multiple of 4");
        let head_size = n_embd / 4;
        let sa_heads = MultiHeadSelfAttention::new(vs / "sa_heads", seq_len, n_embd, head_size, 4);

        let ffwd = FeedForward::new(vs / "ffwd", n_embd);

        let lm_head = nn::linear(vs / "lm_head", n_embd, vocab_size, Default::default());

        Self {
            token_embedding,
            position_embedding,
            lm_head,
            sa_heads,
            ffwd,
            n_embd,
            vocab_size,
            seq_len,
        }
    }
}

impl nn::ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.apply(&self.net).relu()
    }
}

impl nn::ModuleT for NanoGpt {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // FUTURE(ssoudan) we only need t here - we only support t = seq_len - could be made more static
        let (b, t) = xs.size2().unwrap();

        // tok_emb
        let tok_emb = xs.apply(&self.token_embedding); // [batch_size, seq_len, n_embd]

        // pos_emb
        let device = xs.device();
        let pos = Tensor::arange_start(0, t, (tch::Kind::Int64, device)); // [t]
        let pos = pos.unsqueeze(0); // [1, t]
        assert_eq!(pos.size(), &[1, t]);

        let pos_emb = pos.apply(&self.position_embedding);
        let pos_emb = pos_emb.view([t, self.n_embd]); // [seq_len, n_embd]
        assert_eq!(pos_emb.size(), &[t, self.n_embd]);

        // residual connection
        let x = tok_emb + pos_emb; // [batch_size, seq_len, n_embd]
        assert_eq!(x.size(), &[b, t, self.n_embd]);

        // sa heads
        let x = self.sa_heads.forward_t(&x, train); // [batch_size, seq_len, n_embd]
        assert_eq!(x.size(), &[b, t, self.n_embd]);

        // ffwd
        let x = x.apply(&self.ffwd); // [batch_size, seq_len, n_embd]
        assert_eq!(x.size(), &[b, t, self.n_embd]);

        // lm head
        let x = x.apply(&self.lm_head); // [batch_size, seq_len, vocab_size]
        assert_eq!(x.size(), &[b, t, self.vocab_size]);
        x
    }
}

impl LMModel for NanoGpt {
    fn generate(&self, xs: Tensor, max_len: i64) -> Tensor {
        let (_b, t) = xs.size2().unwrap();
        assert_eq!(t, self.seq_len);

        let mut xs = xs;
        for _ in 0..max_len {
            // take the last logits
            let logits = self.forward_t(&xs, false).i((.., -1, ..));
            //  apply softmax to get the probabilities of the next token
            let probs = logits.softmax(-1, tch::Kind::Float);
            // sample the next token
            let next_token = probs.multinomial(1, true);
            xs = Tensor::cat(&[xs, next_token], 1);
            // crop the sequence to the maximum length, starting from the end if needed
            // FUTURE(ssoudan) better way?
            let (_b, t) = xs.size2().unwrap();
            if t > self.seq_len {
                xs = xs.i((.., (t - self.seq_len)..));
            }
            assert_eq!(xs.size()[1], self.seq_len);
        }
        xs
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
