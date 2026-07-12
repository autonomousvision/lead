#!/bin/bash

HOSTNAME=$(hostname)
if [ "$HOSTNAME" = "galvani-xbm002" ]; then
    # On avg-a100-1, connect to avg-a100-2 via internal IP
    parallel_transfer download long@192.168.222.237 /home/long/Desktop/code/lead "$1"
else
    # From local machine, use SSH config alias
    parallel_transfer download avg-a100-2 /home/long/Desktop/code/lead "$1"
fi
