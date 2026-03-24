# Generate renders for spin values from 0 to 0.9 in steps of 0.1

OUTPUT_DIR="angle_sweep"
mkdir -p "$OUTPUT_DIR"

CLI="./build/Release/grrt-cli.exe"

for angle in 1 10 20 30 40 50 60 70 80 90; do
    output="$OUTPUT_DIR/angle_${angle}"
    echo "Rendering angle=$angle -> $output"
    "$CLI" --metric kerr --spin 0.998 --observer-r 1000 --mass 100 --disk-outer 1000 --observer-theta "$angle" --backend cuda --output "$output"
done

echo "Done. Output in $OUTPUT_DIR/"
