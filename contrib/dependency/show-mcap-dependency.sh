#!/bin/bash
#
# Print the commands that install the compression libraries a BUILD_MCAP
# build links: lz4 and zstd.  MCAP stores its chunks compressed with one of
# the two, and the MCAP reader decompresses them with the system libraries
# rather than a vendored copy.
#
# The script installs nothing, because installing a dependency needs your
# review and consent.  It prints the commands for the detected platform, and
# you run the ones you want.  Nothing else in solvcon uses lz4 or zstd, so a
# build with BUILD_MCAP=OFF needs neither.
#
# Usage:
#   ./show-mcap-dependency.sh

set -e

if [ "$#" -ne 0 ]; then
    echo "usage: $(basename "$0")" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin)
        CMDS=("brew install lz4 zstd")
        ;;
    Linux)
        CMDS=("sudo apt-get update"
              "sudo apt-get install -y liblz4-dev libzstd-dev")
        ;;
    MINGW*|MSYS*|CYGWIN*)
        CMDS=("vcpkg install lz4:x64-windows zstd:x64-windows")
        ;;
    *)
        echo "unsupported platform: $(uname -s)" >&2
        echo "Install the lz4 and zstd development packages by hand." >&2
        exit 1
        ;;
esac

echo "solvcon BUILD_MCAP needs the lz4 and zstd development packages."
echo "Review these commands and run the ones you want:"
echo
for cmd in "${CMDS[@]}"; do
    echo "  ${cmd}"
done

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
