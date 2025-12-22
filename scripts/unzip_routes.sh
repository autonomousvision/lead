#!/bin/bash

# Unzip routes from data/carla_leaderboard2/zip to data/carla_leaderboard2/data

SOURCE_DIR="data/carla_leaderboard2/zip"
TARGET_DIR="data/carla_leaderboard2/data"
TEMP_DIR="data/carla_leaderboard2/temp_unzip"

# Create directories if they don't exist
mkdir -p "$TARGET_DIR"
mkdir -p "$TEMP_DIR"

# Find and unzip all zip files
echo "Unzipping routes from $SOURCE_DIR to $TARGET_DIR..."

count=0
for zip_file in "$SOURCE_DIR"/*.zip; do
    if [ -f "$zip_file" ]; then
        echo "Unzipping: $(basename "$zip_file")"

        # Unzip to temp directory
        unzip -q "$zip_file" -d "$TEMP_DIR"

        # Find the deeply nested content and move it to target
        # Look for the pattern: data/carla_leaderboad2_v8/results/data/sensor_data/data/*/
        nested_path=$(find "$TEMP_DIR" -type d -path "*/data/carla_leaderboad2_v8/results/data/sensor_data/data/*" -maxdepth 10 | head -n 1)

        if [ -n "$nested_path" ]; then
            # Move each scenario folder (e.g., Accident) directly to target
            for scenario_dir in "$TEMP_DIR"/data/carla_leaderboad2_v8/results/data/sensor_data/data/*; do
                if [ -d "$scenario_dir" ]; then
                    mv "$scenario_dir" "$TARGET_DIR/"
                    echo "  Moved: $(basename "$scenario_dir")"
                fi
            done
        fi

        # Clean up temp directory
        rm -rf "$TEMP_DIR"
        mkdir -p "$TEMP_DIR"

        count=$((count + 1))
    fi
done

# Remove temp directory
rm -rf "$TEMP_DIR"

echo "Done! Unzipped $count route(s)."
