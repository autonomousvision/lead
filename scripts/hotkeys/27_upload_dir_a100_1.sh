#!/bin/bash

HOSTNAME=$(hostname)
if [ "$HOSTNAME" = "galvani-xbm037" ]; then
    # On avg-a100-2, connect to avg-a100-1 via internal IP
    parallel_transfer upload long@192.168.222.202 /home/long/Desktop/code/lead "$1"
else
    # From local machine, use SSH config alias
    parallel_transfer upload avg-a100-1 /home/long/Desktop/code/lead "$1"
fi
