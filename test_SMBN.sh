#!/usr/bin/env sh
set -e
./caffe-bias/build/tools/caffe test --model=./prototxt_files/indian/test.prototxt --weights=./snapshot/indian/_iter_20000.caffemodel -iterations=9220 -gpu=0   # for the Indian Pines image

