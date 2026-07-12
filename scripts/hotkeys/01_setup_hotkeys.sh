#!/bin/bash

shopt -s globstar

rm *.sh
rm *.py

ln -s scripts/hotkeys/* .
