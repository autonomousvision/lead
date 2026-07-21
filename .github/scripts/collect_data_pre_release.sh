#!/bin/bash
# Pre-release tier: one route per scenario type, spread across both towns.
exec bash "$(dirname "$0")/collect_data.sh" \
    50x36_Town13/AccidentTwoWays/1153_0.xml \
    50x36_Town13/BlockedIntersection/1016_0.xml \
    50x36_Town13/ConstructionObstacleTwoWays/1128_0.xml \
    50x36_Town13/HardBreakRoute/1015_0.xml \
    50x38_Town12/Accident/10_0.xml \
    50x38_Town12/CrossingBicycleFlow/136_0.xml \
    50x38_Town12/DynamicObjectCrossing/2410_0.xml \
    50x38_Town12/OppositeVehicleRunningRedLight/3717_0.xml \
    50x38_Town12/ParkedObstacle/4006_0.xml \
    50x38_Town12/PedestrianCrossing/2862_0.xml
