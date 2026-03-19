#!/bin/bash
# Generate renders for spin values from 0 to 0.9 in steps of 0.1

OUTPUT_DIR="spin_sweep"
mkdir -p "$OUTPUT_DIR"

CLI="./build/Release/grrt-cli.exe"

for spin in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    output="$OUTPUT_DIR/spin_${spin}.hdr"
    echo "Rendering spin=$spin -> $output"
    "$CLI" --metric kerr --spin "$spin" --observer-r 50 --backend cuda --output "$output"
done

echo "Done. Output in $OUTPUT_DIR/"
