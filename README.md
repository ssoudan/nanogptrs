# nanoGPT(rs)

This is a Rust implementation of the nanoGPT model from Andrej Karpathy's YT
video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=12s

With some help from: https://github.com/LaurentMazare/tch-rs/blob/main/examples/min-gpt/main.rs and
https://github.com/karpathy/nanoGPT/blob/master/model.py.

# Setup

Create micromamba environment (or conda):

```bash
micromamba env create -f environment.yml
```

Activate environment:

```bash
micromamba activate nanogptrs
```

```bash
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib/:$LD_LIBRARY_PATH
```

# Run - generate from pretrained GPT-2

```bash
./data/download.sh
./models/download.sh gpt2
cargo run --release -- --device=cuda --restore-from models/gpt2/model.safetensors generate --max-len 32 --prompt "Once upon a time" gpt2
```

# Run - train nano-gpt

```bash
cargo run --release -- --device=cuda train --n-epochs=3 --final-checkpoint-path=models/nanogptrs.safetensors nano-gpt
```

Should eventually (~5h on my Titan XP) produce something like this:

```
DUCHESS OF YORK:
Here comes already.

EXTOLY:
O, by the means of your crown?

KING HENRY VI:
Brother, that my lord, change thou givest queen.

KING RICHARD II:
Mine honour, because I am advertised
The queen our is not your voice. Would thy sight
Next Rome, among, insible express to dictliffe:
For ere for goings
Abova drunking redel her food pain soul to every it.

QUEEN MARGARET:
I took! O, if you so, good and the Montague of slave,
That he's breathing which holy a holy brats.
```
