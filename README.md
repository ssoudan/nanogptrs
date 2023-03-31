# nanoGPT(rs)

This is a Rust implementation of the nanoGPT model from Andrej Karpathy's YT 
video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=12s

With some help from: https://github.com/LaurentMazare/tch-rs/blob/main/examples/min-gpt/main.rs and 
https://github.com/karpathy/nanoGPT/blob/master/model.py. 

# Setup
Create conda environment:

```bash
conda env create -f environment.yml
``` 

Activate conda environment:

```bash
rm -f torch # remove symlink if it exists
conda activate nanoGPT
TORCH_DIR=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
ln -sf $TORCH_DIR torch
ls -l torch
```

# Run

```bash
cargo run --release -- --device=cuda --n-epochs=3 nano-gpt
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