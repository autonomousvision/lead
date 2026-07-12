#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

SESSION_NAME="screen_session_$(date +%s)"

screen -dmS "$SESSION_NAME" bash -c "$*"
echo "Started command in screen session: $SESSION_NAME with command '$*'"
