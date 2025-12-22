download_meta_data() {
    mkdir -p "$1"

    rsync -avz --info=progress2 \
        --include="*/" \
        --include="*.pkl" \
        --include="results.json" \
        --exclude="*" \
        tcml3:/home/nguyen/lead/"$1"/ "$1/"
}

while true; do
    while read -r line; do
        download_meta_data "$line" &
        sleep 1
    done < <(cat <<EOF | shuf
data/carla_leaderboad2_v14/results/data/sensor_data/data/NonSignalizedJunctionLeftTurnEnterFlow
data/carla_leaderboad2_v14/results/data/sensor_data/data/SignalizedJunctionLeftTurnEnterFlow
data/carla_leaderboad2_v14/results/data/sensor_data/data/PedestrianCrossing
data/carla_leaderboad2_v14/results/data/sensor_data/data/noScenarios
data/carla_leaderboad2_v14/results/data/sensor_data/data/SignalizedJunctionRightTurn
data/carla_leaderboad2_v14/results/data/sensor_data/data/ParkingCrossingPedestrian
data/carla_leaderboad2_v14/results/data/sensor_data/data/AccidentTwoWays
data/carla_leaderboad2_v14/results/data/sensor_data/data/ParkedObstacleTwoWays
data/carla_leaderboad2_v14/results/data/sensor_data/data/DynamicObjectCrossing
data/carla_leaderboad2_v14/results/data/sensor_data/data/ParkedObstacle
data/carla_leaderboad2_v14/results/data/sensor_data/data/VehicleTurningRoute
data/carla_leaderboad2_v14/results/data/sensor_data/data/ConstructionObstacle
data/carla_leaderboad2_v14/results/data/sensor_data/data/InvadingTurn
data/carla_leaderboad2_v14/results/data/sensor_data/data/SignalizedJunctionLeftTurn
data/carla_leaderboad2_v14/results/data/sensor_data/data/OppositeVehicleTakingPriority
data/carla_leaderboad2_v14/results/data/sensor_data/data/HighwayExit
data/carla_leaderboad2_v14/results/data/sensor_data/data/NonSignalizedJunctionLeftTurn
data/carla_leaderboad2_v14/results/data/sensor_data/data/OppositeVehicleRunningRedLight
data/carla_leaderboad2_v14/results/data/sensor_data/data/EnterActorFlowV2
data/carla_leaderboad2_v14/results/data/sensor_data/data/InterurbanAdvancedActorFlow
data/carla_leaderboad2_v14/results/data/sensor_data/data/InterurbanActorFlow
data/carla_leaderboad2_v14/results/data/sensor_data/data/ParkingExit
data/carla_leaderboad2_v14/results/data/sensor_data/data/BlockedIntersection
data/carla_leaderboad2_v14/results/data/sensor_data/data/HazardAtSideLaneTwoWays
data/carla_leaderboad2_v14/results/data/sensor_data/data/ControlLoss
data/carla_leaderboad2_v14/results/data/sensor_data/data/PriorityAtJunction
data/carla_leaderboad2_v14/results/data/sensor_data/data/NonSignalizedJunctionRightTurn
data/carla_leaderboad2_v14/results/data/sensor_data/data/YieldToEmergencyVehicle
data/carla_leaderboad2_v14/results/data/sensor_data/data/HardBreakRoute
data/carla_leaderboad2_v14/results/data/sensor_data/data/VehicleTurningRoutePedestrian
data/carla_leaderboad2_v14/results/data/sensor_data/data/Accident
data/carla_leaderboad2_v14/results/data/sensor_data/data/ConstructionObstacleTwoWays
data/carla_leaderboad2_v14/results/data/sensor_data/data/EnterActorFlow
data/carla_leaderboad2_v14/results/data/sensor_data/data/ParkingCutIn
data/carla_leaderboad2_v14/results/data/sensor_data/data/VehicleOpensDoorTwoWays
data/carla_leaderboad2_v14/results/data/sensor_data/data/StaticCutIn
data/carla_leaderboad2_v14/results/data/sensor_data/data/HazardAtSideLane
data/carla_leaderboad2_v14/results/data/sensor_data/data/HighwayCutIn
data/carla_leaderboad2_v14/results/data/sensor_data/data/CrossingBicycleFlow
data/carla_leaderboad2_v14/results/data/sensor_data/data/MergerIntoSlowTraffic
data/carla_leaderboad2_v14/results/data/sensor_data/data/MergerIntoSlowTrafficV2
data/carla_leaderboad2_v14/results/data/sensor_data/data/RedLightWithoutLeadVehicle
data/carla_leaderboad2_v14/results/data/sensor_data/data/T_Junction
data/carla_leaderboad2_v14/results/data/sensor_data/data/CrossJunctionDefectTrafficLight
EOF
    )
    wait
    sleep 10
done
