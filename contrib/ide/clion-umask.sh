#!/bin/bash
#
# Launch CLion from the macOS command line with a custom umask.
#
# Usage: clion-umask.sh -m UMASK [-a CLION_APP] [-d DATA_DIR] [-- ARGS...]
#
#   -m UMASK      Octal umask (required), e.g. 0002. Recommended for a
#                 checkout shared by a group: new files get mode 664 and new
#                 directories 775, so the group can write them.
#   -a CLION_APP  CLion.app to launch (default /Applications/CLion.app, or
#                 $CLION_APP).
#   -d DATA_DIR   Data for the separate instance (default
#                 ~/.local/share/clion-umask, or $CLION_UMASK_DATA).
#   -- ARGS       Passed to CLion, e.g. a project path.
#
# `open -a CLion` ignores the shell's umask, so the script runs the binary in
# the app bundle directly. A running CLion keeps its own umask, so when the
# current user already runs one, the script starts a separate instance with
# its settings in DATA_DIR/<umask>. On first use it asks whether to copy the
# installed version's settings (needs a terminal; sign-in is not copied).
#

set -euo pipefail

UMASK=""
CLION_APP=${CLION_APP:-/Applications/CLion.app}
DATA_DIR=${CLION_UMASK_DATA:-$HOME/.local/share/clion-umask}

usage() {
    sed -n '3,/^$/p' "$0" | sed -e 's/^# \{0,1\}//' -e '/^#$/d' >&2
    exit "${1:-1}"
}

while getopts "m:a:d:h" opt; do
    case "$opt" in
        m) UMASK=$OPTARG ;;
        a) CLION_APP=$OPTARG ;;
        d) DATA_DIR=$OPTARG ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift

[[ "$(uname -s)" == "Darwin" ]] || {
    echo "macOS is required" >&2
    exit 1
}

[[ -n "$UMASK" ]] || {
    echo "-m UMASK is required, for example -m 0002" >&2
    usage 1
}

[[ "$UMASK" =~ ^0?[0-7]{3}$ ]] || {
    echo "invalid umask: $UMASK (expect octal like 0002 or 022)" >&2
    exit 1
}

CLION_BIN="$CLION_APP/Contents/MacOS/clion"
[[ -x "$CLION_BIN" ]] || {
    echo "CLion executable not found: $CLION_BIN (see -a)" >&2
    exit 1
}

copy_settings() {
    local dest=$1 base ver src="" ans
    base="$HOME/Library/Application Support/JetBrains"
    ver=$(defaults read "$CLION_APP/Contents/Info" \
        CFBundleShortVersionString 2>/dev/null | cut -d. -f1,2) || true
    [[ -n "$ver" && -d "$base/CLion$ver" ]] && src="$base/CLion$ver"
    if [[ -z "$src" ]]; then
        src=$(ls -d "$base"/CLion[0-9]* 2>/dev/null | tail -1) || true
    fi
    [[ -n "$src" && -d "$src" ]] || return 0
    [[ -t 0 ]] || {
        echo "No terminal to ask on; starting with empty settings" >&2
        return 0
    }
    read -r -p "Copy CLion settings from $src to $dest? [y/N] " ans
    [[ "$ans" =~ ^[Yy] ]] && cp -R "$src" "$dest"
    return 0
}

umask "$UMASK"
UMASK_OCT=$(umask)
PID_FILE=""

if pgrep -u "$(id -u)" -f "$CLION_BIN" >/dev/null; then
    inst_dir="$DATA_DIR/$UMASK_OCT"
    mkdir -p "$inst_dir"
    [[ -e "$inst_dir/config" ]] || copy_settings "$inst_dir/config"
    # A copy carries the running CLion's lock files, and a crashed instance
    # leaves its own; either makes CLion believe a live process owns the
    # directory. Keep the locks only while the instance started here is alive.
    if [[ -r "$inst_dir/clion.pid" ]] &&
        kill -0 "$(<"$inst_dir/clion.pid")" 2>/dev/null; then
        echo "Reusing the running instance in $inst_dir"
    else
        rm -f "$inst_dir"/{config,system}/{.lock,.port}
        PID_FILE="$inst_dir/clion.pid"
    fi
    cat >"$inst_dir/idea.properties" <<EOF
idea.config.path=$inst_dir/config
idea.plugins.path=$inst_dir/config/plugins
idea.system.path=$inst_dir/system
idea.log.path=$inst_dir/log
EOF
    export CLION_PROPERTIES="$inst_dir/idea.properties"
    echo "CLion is running; starting a separate instance in $inst_dir"
fi

echo "Launching $CLION_APP with umask $UMASK_OCT"
(
    nohup "$CLION_BIN" "$@" </dev/null >/dev/null 2>&1 &
    [[ -z "$PID_FILE" ]] || echo $! >"$PID_FILE"
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
