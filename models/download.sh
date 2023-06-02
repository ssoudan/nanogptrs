#!/bin/bash

# Models from args
MODELS=$@
[ -z "$MODELS" ] && MODELS="gpt2 gpt2-medium"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

function download_if_not_exists() {
    if [ ! -f $2 ]; then
        wget -O $2 $1
    else
        echo "$2 already exists, skipping download"
    fi
}

for g in $MODELS;
do
    mkdir -p $DIR/$g
    download_if_not_exists https://huggingface.co/$g/resolve/main/model.safetensors $DIR/$g/model.safetensors
    download_if_not_exists https://huggingface.co/$g/resolve/main/tokenizer.json $DIR/$g/tokenizer.json
done

echo "Done!"
