#!/bin/bash
# Run from the repo root, regardless of the invoking directory.
cd "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")/../.."

python3 src/lead/webapp/app.py
