#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
CONFIG=Release
CUDA=OFF

usage() {
    echo "Usage: $0 [clean|debug|cuda|help]"
    echo "  clean  - Remove build directory"
    echo "  debug  - Build in Debug configuration"
    echo "  cuda   - Build with CUDA backend enabled"
    echo "  help   - Show this message"
}

for arg in "$@"; do
    case "$arg" in
        clean)
            echo "Cleaning build directory..."
            rm -rf "$BUILD_DIR"
            echo "Done."
            exit 0
            ;;
        debug)
            CONFIG=Debug
            ;;
        cuda)
            CUDA=ON
            ;;
        help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            exit 1
            ;;
    esac
done

echo "Configuring ($CONFIG, CUDA=$CUDA)..."
cmake -B "$BUILD_DIR" -S "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE="$CONFIG" \
    -DGRRT_ENABLE_CUDA="$CUDA"

echo "Building..."
cmake --build "$BUILD_DIR" --config "$CONFIG" -- -j"$(nproc)"

echo ""
echo "Build complete: $BUILD_DIR/grrt-cli"
