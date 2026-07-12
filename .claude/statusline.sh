#!/bin/bash
# .claude/statusline.sh
#
# Status line for Claude Code. Reads the session JSON on stdin and
# prints a single line with model, project name, branch (with `*` if
# dirty), context-window usage, token counts, estimated cost, and
# Claude.ai subscription usage (5-hour and weekly windows) when the
# account exposes them.
# Falls back gracefully if `jq` is not installed.

input=$(cat)

if command -v jq >/dev/null 2>&1; then
    model=$(printf '%s' "$input" | jq -r '.model.display_name // "?"')
    dir=$(printf '%s' "$input" | jq -r '.workspace.current_dir // ""')
    used=$(printf '%s' "$input" | jq -r '.context_window.used_percentage // empty')
    total_in=$(printf '%s' "$input" | jq -r '.context_window.total_input_tokens // empty')
    total_out=$(printf '%s' "$input" | jq -r '.context_window.total_output_tokens // empty')
    five=$(printf '%s' "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')
    week=$(printf '%s' "$input" | jq -r '.rate_limits.seven_day.used_percentage // empty')
else
    model="?"
    dir=""
    used=""
    total_in=""
    total_out=""
    five=""
    week=""
fi

[ -z "$dir" ] && dir="$PWD"
proj=$(basename "$dir")
branch=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo no-branch)
dirty=""
if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
    dirty="*"
fi

ctx_part=$(printf ' | ctx %s%%' "$(printf '%.0f' "$used")")

tok_part=""
if [ -n "$total_in" ] && [ -n "$total_out" ]; then
    tok_part=$(printf ' | in %s out %s' "$total_in" "$total_out")
fi

# Estimated session cost, derived from token counts and USD per million tokens
# by models. These are list prices and ignore prompt-cache discounts, so treat
# the figure as an upper bound.
cost_part=""
if [ -n "$total_in" ] && [ -n "$total_out" ]; then
    case "$model" in
        *Opus*)   in_rate=15; out_rate=75 ;;
        *Sonnet*) in_rate=3;  out_rate=15 ;;
        *Haiku*)  in_rate=1;  out_rate=5  ;;
        *)        in_rate="";  out_rate="" ;;
    esac
    if [ -n "$in_rate" ]; then
        cost_part=$(awk -v i="$total_in" -v o="$total_out" \
            -v ir="$in_rate" -v or="$out_rate" \
            'BEGIN { printf " | est $%.2f", (i*ir + o*or) / 1000000 }')
    fi
fi

# Claude.ai subscription usage windows, appended after the estimated cost.
# Each segment is shown only when the account exposes that window (present for
# subscribers after the first API response); nothing is added when none are
# set.
usage_part=""
[ -n "$five" ] && usage_part=$(printf '%s | 5h %s%%' "$usage_part" "$(printf '%.0f' "$five")")
[ -n "$week" ] && usage_part=$(printf '%s | wk %s%%' "$usage_part" "$(printf '%.0f' "$week")")

printf '%s | %s | %s%s%s%s%s%s' \
    "$model" "$proj" "$branch" "$dirty" "$ctx_part" "$tok_part" "$cost_part" "$usage_part"
