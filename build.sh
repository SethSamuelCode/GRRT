#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
CONFIG=Release
CUDA=OFF

# Windows uses the multi-config Visual Studio generator (binaries land in
# build/<Config>/ with a .exe suffix); Linux uses a single-config Makefile
# generator (binary lands directly in build/, no suffix).
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) PLATFORM=windows ;;
    *)                    PLATFORM=linux ;;
esac

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
cmake --build "$BUILD_DIR" --config "$CONFIG" --parallel "$(nproc)"

if [ "$PLATFORM" = windows ]; then
    CLI_PATH="$BUILD_DIR/$CONFIG/grrt-cli.exe"
else
    CLI_PATH="$BUILD_DIR/grrt-cli"
fi

echo ""
echo "Build complete: $CLI_PATH"
