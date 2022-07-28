#!/usr/bin/env bash

./build.sh

docker save PLORAS | gzip -c > PLORAS.tar.gz
