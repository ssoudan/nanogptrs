#!/usr/bin/env bash

set -e

exec tensorboard --logdir=summaries --port=6006 --host=0.0.0.0
