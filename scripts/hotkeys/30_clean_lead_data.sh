#!/bin/bash

find "$(dotenv PY123D_DATA_ROOT)" -mindepth 1 -maxdepth 1 ! -name maps -exec rm -rf {} +
