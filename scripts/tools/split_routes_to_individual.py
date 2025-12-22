#!/usr/bin/env python3
"""Split route XML files into individual route files."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List


def format_xml_route(route_elem: ET.Element) -> str:
    """Format a single route element as a nicely formatted XML string."""
    # Create root element
    root = ET.Element('routes')
    root.text = '\n   '
    root.tail = '\n'
    
    # Clone the route element
    new_route = ET.Element('route')
    new_route.set('id', route_elem.get('id'))
    new_route.set('town', route_elem.get('town'))
    new_route.text = '\n      '
    new_route.tail = '\n'
    
    # Copy waypoints
    waypoints_elem = route_elem.find('waypoints')
    if waypoints_elem is not None:
        waypoints = ET.SubElement(new_route, 'waypoints')
        waypoints.text = '\n         '
        waypoints.tail = '\n      '
        
        positions = waypoints_elem.findall('position')
        for i, pos in enumerate(positions):
            position = ET.SubElement(waypoints, 'position')
            position.set('x', pos.get('x'))
            position.set('y', pos.get('y'))
            position.set('z', pos.get('z'))
            
            if i < len(positions) - 1:
                position.tail = '\n         '
            else:
                position.tail = '\n      '
    
    # Copy scenarios
    scenarios_elem = route_elem.find('scenarios')
    if scenarios_elem is not None:
        scenarios = ET.SubElement(new_route, 'scenarios')
        
        scenario_list = scenarios_elem.findall('scenario')
        if scenario_list:
            scenarios.text = '\n         '
            scenarios.tail = '\n   '
            
            for i, scen in enumerate(scenario_list):
                scenario = ET.SubElement(scenarios, 'scenario')
                scenario.set('type', scen.get('type'))
                scenario.set('name', scen.get('name'))
                scenario.text = '\n            '
                
                if i < len(scenario_list) - 1:
                    scenario.tail = '\n         '
                else:
                    scenario.tail = '\n      '
                
                # Copy trigger_point
                trigger = scen.find('trigger_point')
                if trigger is not None:
                    trigger_point = ET.SubElement(scenario, 'trigger_point')
                    trigger_point.set('x', trigger.get('x'))
                    trigger_point.set('y', trigger.get('y'))
                    trigger_point.set('z', trigger.get('z'))
                    trigger_point.set('yaw', trigger.get('yaw'))
                    trigger_point.tail = '\n         '
        else:
            # Empty scenarios
            scenarios.tail = '\n   '
    
    root.append(new_route)
    
    # Convert to string
    tree = ET.ElementTree(root)
    return ET.tostring(root, encoding='unicode')


def split_route_file(input_file: Path, output_dir: Path) -> int:
    """Split a route XML file into individual route files."""
    # Parse the input XML
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Get base filename without extension
    base_name = input_file.stem
    
    # Create subdirectory for this file's routes
    file_output_dir = output_dir / input_file.parent.name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each route
    route_count = 0
    for route in root.findall('route'):
        route_id = route.get('id')
        
        # Create output filename: Town06_Scenario1_0.xml
        output_filename = f"{base_name}_{route_id}.xml"
        output_path = file_output_dir / output_filename
        
        # Format and write the XML
        xml_content = format_xml_route(route)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
            f.write(xml_content)
        
        route_count += 1
    
    print(f"Split {input_file.name}: {route_count} routes -> {file_output_dir.name}/")
    return route_count


def main():
    """Main function to split all route files."""
    workspace = Path(__file__).parent.parent.parent
    input_dir = workspace / 'data' / 'routes' / 'lb1_split_2'
    output_dir = workspace / 'data' / 'routes' / 'lb1_split_2_individuals'
    
    # Find all XML files recursively
    xml_files: List[Path] = []
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            xml_files.extend(subdir.glob('*.xml'))
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files to split\n")
    
    total_routes = 0
    for xml_file in sorted(xml_files):
        route_count = split_route_file(xml_file, output_dir)
        total_routes += route_count
    
    print(f"\n✓ Split complete!")
    print(f"  Total files processed: {len(xml_files)}")
    print(f"  Total routes created: {total_routes}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
