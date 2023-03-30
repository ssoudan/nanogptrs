# nanoGPT rs

More or less what this video is about: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=12s 
With some help from: https://github.com/LaurentMazare/tch-rs/blob/main/examples/min-gpt/main.rs
and: https://github.com/karpathy/nanoGPT/blob/master/model.py 

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
