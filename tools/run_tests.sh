#!/bin/sh
# 1. Activate the environment
source activate miniflow
# 2. Run the experiment
py.test -s ./tests/
