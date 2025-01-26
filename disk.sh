#! /bin/bash

# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

TARGET_DIR="$1"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

# Set minimum free space threshold (200MB in KB)
MIN_FREE_SPACE=204800

# Counter for generated files
counter=1

while true; do
    # Get available disk space in KB
    available_space=$(df -k "$TARGET_DIR" | tail -n 1 | awk '{print $4}')
    
    # Check if available space is less than minimum threshold
    if [ "$available_space" -lt "$MIN_FREE_SPACE" ]; then
        echo "Reached minimum free space threshold (100MB). Stopping."
        exit 0
    fi
    
    # Create a file with 1MB of random data
    dd if=/dev/urandom of="$TARGET_DIR/file_$counter" bs=100M count=1 2>/dev/null
    
    echo "Created file_$counter"
    counter=$((counter + 1))
    
    # Small sleep to prevent overwhelming the system
    sleep 0.1
done
