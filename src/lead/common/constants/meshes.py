"""Mesh/type-id groups and static-mesh extent lookup table."""

EMERGENCY_MESHES = {
    "vehicle.dodge.charger_police_2020",
    "vehicle.dodge.charger_police",
    "vehicle.ford.ambulance",
    "vehicle.carlamotors.firetruck",
}

CONSTRUCTION_MESHES = {
    "static.prop.constructioncone",
    "static.prop.trafficwarning",
    "static.prop.warningconstruction",
    "static.prop.warningaccident",
}

# ``traffic.*`` actors that have their own box class and are therefore not
# collected as generic traffic signs.
NON_SIGN_TRAFFIC_TYPES = {"traffic.traffic_light", "traffic.stop"}

BIKER_MESHES = {
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
    "vehicle.bh.crossbike",
}

LOOKUP_TABLE = {
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": [
        2.44619083404541,
        1.115301489830017,
        0.7606233954429626,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": [
        2.5039126873016357,
        1.0485419034957886,
        0.7673624753952026,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": [
        2.2210919857025146,
        0.9388753771781921,
        0.9936029314994812,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": [
        2.6828393936157227,
        0.9732309579849243,
        0.7874829173088074,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": [
        2.782914400100708,
        1.217571496963501,
        1.022573471069336,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": [
        2.3368194103240967,
        1.0011461973190308,
        0.7259762287139893,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": [
        2.3958897590637207,
        1.081725001335144,
        0.7438300848007202,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": [
        2.2763495445251465,
        1.0926425457000732,
        0.8835831880569458,
    ],
    "/Game/Carla/Static/Dynamic/Garden/SM_PlasticTable.SM_PlasticTable": [
        1.241101622581482,
        1.241101622581482,
        1.239898920059204,
    ],
    "/Game/Carla/Static/Dynamic/Garden/SM_PlasticChair.SM_PlasticChair": [
        0.36523768305778503,
        0.37522444128990173,
        0.6356779336929321,
    ],
    "/Game/Carla/Static/Dynamic/Construction/SM_ConstructionCone.SM_ConstructionCone": [
        0.1720348298549652,
        0.1720348298549652,
        0.2928849756717682,
    ],
}

CONSTRUCTION_CONE_BB_SIZE = [0.1720348298549652, 0.1720348298549652]

TRAFFIC_WARNING_BB_SIZE = [1.186714768409729, 1.4352929592132568]
