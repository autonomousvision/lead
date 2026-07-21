#!/bin/bash
# Main tier: a spread over static obstacle, VRU, and intersection scenarios.
exec bash "$(dirname "$0")/collect_data.sh" \
    50x38_Town12/Accident/10_0.xml \
    50x38_Town12/DynamicObjectCrossing/2410_0.xml \
    50x38_Town12/OppositeVehicleRunningRedLight/3717_0.xml \
    50x38_Town12/ParkedObstacle/4006_0.xml \
    50x38_Town12/PedestrianCrossing/2862_0.xml
