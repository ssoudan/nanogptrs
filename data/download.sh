#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

function download_if_not_exists() {
    if [ ! -f $2 ]; then
        wget -O $2 $1
    else
        echo "$2 already exists, skipping download"
    fi
}

mkdir -p $DIR/$g
#download_if_not_exists https://github.com/matthewreagan/WebstersEnglishDictionary/blob/master/dictionary_compact.json $DIR/dictionary_compact.json
download_if_not_exists https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt $DIR/input.txt
#download_if_not_exists https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt $DIR/TinyStoriesV2-GPT4-train.txt
#download_if_not_exists https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt $DIR/TinyStoriesV2-GPT4-valid.txt

echo "Done!"
