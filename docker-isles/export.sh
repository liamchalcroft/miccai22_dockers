#!/usr/bin/env bash

./build.sh

docker save threshold_model | gzip -c > threshold_model.tar.gz
