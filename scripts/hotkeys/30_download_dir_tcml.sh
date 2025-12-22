#!/bin/bash

mkdir -p "$1"

rsync -avz --info=progress2 tcml3:/home/nguyen/lead/$1/ "$1"/
