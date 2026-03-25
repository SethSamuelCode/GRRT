#!/usr/bin/env bash
# validate_volumetric.sh — End-to-end validation for the volumetric accretion disk.
# Runs thin-disk regression, volumetric renders at various angles, and a spin sweep.
# Usage: bash scripts/validate_volumetric.sh [path-to-grrt-cli]

set -euo pipefail

CLI="${1:-./build/Release/grrt-cli.exe}"
OUTDIR="validation_output"
WIDTH=256
HEIGHT=256
COMMON="--width $WIDTH --height $HEIGHT"

mkdir -p "$OUTDIR"

echo "=== Volumetric Disk Validation ==="
echo "CLI: $CLI"
echo "Output: $OUTDIR/"
echo ""

# Step 1: Thin disk regression (no --disk-volumetric)
echo "[Step 1] Thin disk regression render..."
$CLI --metric kerr --spin 0.998 --observer-r 50 $COMMON \
     --output "$OUTDIR/regression"
echo "  -> OK"
echo ""

# Step 2: CPU/CUDA validation (--validate compares both backends)
echo "[Step 2] CPU/CUDA volumetric validation..."
$CLI --metric kerr --spin 0.998 --observer-r 50 --disk-volumetric --validate $COMMON \
     --output "$OUTDIR/vol_validate"
echo "  -> OK"
echo ""

# Step 3: Face-on vs edge-on
echo "[Step 3a] Face-on (theta=15 deg)..."
$CLI --metric kerr --spin 0.998 --observer-r 50 --observer-theta 15 \
     --disk-volumetric $COMMON --output "$OUTDIR/vol_face_on"
echo "  -> OK"

echo "[Step 3b] Edge-on (theta=80 deg)..."
$CLI --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 \
     --disk-volumetric $COMMON --output "$OUTDIR/vol_edge_on"
echo "  -> OK"
echo ""

# Step 4: Spin sweep
echo "[Step 4] Spin sweep..."
for spin in 0.0 0.3 0.6 0.9 0.998; do
    echo "  spin=$spin"
    $CLI --metric kerr --spin "$spin" --observer-r 50 --disk-volumetric $COMMON \
         --output "$OUTDIR/vol_spin_${spin}"
done
echo "  -> OK"
echo ""

echo "=== All validation renders completed successfully ==="
echo "Output images are in $OUTDIR/"
