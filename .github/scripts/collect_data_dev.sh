#!/bin/bash
# Smoke tier for dev branches: one static-obstacle and one VRU scenario.
exec bash "$(dirname "$0")/collect_data.sh" \
    50x38_Town12/Accident/10_0.xml \
    50x38_Town12/PedestrianCrossing/2862_0.xml
