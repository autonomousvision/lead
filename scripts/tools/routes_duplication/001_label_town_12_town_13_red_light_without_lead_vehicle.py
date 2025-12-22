import xml.etree.ElementTree as ET
import os
import shutil

def function(town):
    prefix = "50x38" if town == "Town12" else "50x36"
    src_path = f"data/routes/{prefix}_{town}/SignalizedJunctionLeftTurn"
    tgt_path = f"data/routes/scenario_swapped_routes/{town}/SignalizedJunctionLeftTurn/RedLightWithoutLeadVehicle"

    for file in os.listdir(src_path):
        if file.endswith(".xml"):
            src_file = os.path.join(src_path, file)
            tgt_file = os.path.join(tgt_path, file)

            # Create target directory
            os.makedirs(os.path.dirname(tgt_file), exist_ok=True)

            # Parse XML
            tree = ET.parse(src_file)
            root = tree.getroot()

            # Update all scenario names and types
            for scenario in root.findall(".//scenario"):
                if scenario.attrib.get("type") == "SignalizedJunctionLeftTurn":
                    scenario.attrib["type"] = "RedLightWithoutLeadVehicle"
                if scenario.attrib.get("name", "").startswith("SignalizedJunctionLeftTurn"):
                    # Replace the prefix but keep the suffix (e.g., _1, _2)
                    old_name = scenario.attrib["name"]
                    suffix = old_name.replace("SignalizedJunctionLeftTurn", "")
                    scenario.attrib["name"] = f"RedLightWithoutLeadVehicle{suffix}"

            # Write modified XML to target
            tree.write(tgt_file, encoding="utf-8", xml_declaration=True)
            print(f"Copied and renamed scenarios in {file} to {tgt_path}")

if __name__ == "__main__":
    function("Town12")
    function("Town13")
