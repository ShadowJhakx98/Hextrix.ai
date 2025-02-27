#!/bin/bash

# Define the base directory
BASE_DIR="project_root"

# Create the directory structure
mkdir -p $BASE_DIR/{docs,src,modules,quantum,assets/{textures,shaders},tests,configs,data/{images,logs,backups,audio,temp}}

echo "Directory structure created:"
tree -d $BASE_DIR
