#!/usr/bin/env sh
set -e

./caffe-bias/build/tools/caffe train --solver=./prototxt_files/indian/solver.prototxt $@  # for training the SMBN on the Indian Pines image

