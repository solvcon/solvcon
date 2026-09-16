#!/usr/bin/env python3
"""Block AI attribution in commit messages and GitHub prose.

Shell-command hook shared by Claude Code (PreToolUse on Bash), Codex
(PreToolUse on Bash), and Cursor (beforeShellExecution).  It reads the
calling agent's hook JSON on stdin and finds each git or gh write in the
command line.  It scans the words of that write, and any message file
the write names, for prose that credits an AI agent.  Commit and PR
prose in solvcon is human-authored and never advertises the tool that
produced it (see CLAUDE.md).

Claude Code and Codex read a block from exit status 2 plus stderr.
Cursor reads a JSON verdict on stdout.
"""

import json
import os
import re
import shlex
import sys

# Characters that end a simple command.  A newline outside quotes starts
# a new command; a newline inside a quoted message does not.
PUNCTUATION = "();<>|&{}\n"

WRITES = {
    "git": {"commit", "merge", "tag", "notes"},
    "gh": {"create", "new", "edit", "comment", "review", "merge", "close",
           "reopen"},
}

AGENTS = r"\b(?:claude|codex|cursor|copilot|chatgpt|gpt|gemini|llama|" \
         r"anthropic|openai|devin)\b"

PATTERNS = [
    (re.compile(r"^\s*(?:co-authored-by|assisted-by|co-created-by"
                r"|generated-by|signed-off-by):.*" + AGENTS, re.I | re.M),
     "attribution trailer naming an AI agent"),
    (re.compile(r"(?:generated|made|created|written|authored|assisted)"
                r"\s+(?:with|by)\s+\[?(?:" + AGENTS
                + r"|\bAI\b|artificial intelligence)", re.I),
     "credit line naming an AI agent"),
    (re.compile(r"noreply@anthropic\.com|claude\.com/claude-code", re.I),
     "AI agent address or tool link"),
    (re.compile("\U0001f916"), "robot emoji"),
]


def read_input():
    try:
        data = json.loads(sys.stdin.read())
    except ValueError:
        return None, "", ""
    if not isinstance(data, dict):
        return None, "", ""

    tool_input = data.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    command = tool_input.get("command") or data.get("command")
    cwd = tool_input.get("cwd") or data.get("cwd")
    if not isinstance(command, str):
        command = ""
    if not isinstance(cwd, str):
        cwd = os.getcwd()
    return data.get("hook_event_name"), command, cwd


def simple_commands(command):
    """Split a command line into argv lists, one per simple command."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=PUNCTUATION)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()

    out, current = [], []
    for token in tokens:
        if token and set(token) <= set(PUNCTUATION):
            out.append(current)
            current = []
        else:
            current.append(token)
    out.append(current)
    return [argv for argv in out if argv]


def publishes_prose(argv):
    """True when argv runs git or gh with a write subcommand after it.

    The hook matches any later word, not the exact subcommand position.
    A wrapper (`env -u VAR`, `if ...; then`) or a global option
    (`git -C dir`, `gh --repo owner/name`) then needs no parser.
    """
    for index, word in enumerate(argv):
        writes = WRITES.get(os.path.basename(word))
        if writes and writes.intersection(argv[index + 1:]):
            return True
    return False


def prose(argv, cwd, command):
    """The text argv publishes: its own words, plus any file it names.

    A file named `-` is standard input, which an agent feeds from a
    heredoc on the same command line, so the whole line is scanned.
    """
    texts = list(argv)
    for index, arg in enumerate(argv):
        if not arg.startswith("-") or ("file" not in arg
                                       and not arg.startswith("-F")):
            continue
        if "=" in arg:
            value = arg.split("=", 1)[1]
        elif arg.startswith("-F") and len(arg) > 2:
            value = arg[2:]
        else:
            value = argv[index + 1] if index + 1 < len(argv) else ""
        if value == "-":
            texts.append(command)
            continue
        if not value:
            continue
        try:
            with open(os.path.join(cwd, value), errors="replace") as stream:
                texts.append(stream.read())
        except OSError:
            continue
    return "\n".join(texts)


def scan(text):
    hits = []
    for pattern, label in PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append("%s -- %s" % (label, match.group(0).strip()[:80]))
    return hits


def report(reasons):
    return ("Hook violation: AI attribution in a commit message or GitHub "
            "prose.\n"
            + "\n".join("  " + reason for reason in reasons) + "\n"
            "solvcon commit messages and PR bodies are human-authored: drop "
            "the\nco-author trailer, the \"Generated with\" line, and the "
            "robot emoji,\nthen rerun (see CLAUDE.md, \"Pull Request "
            "Guidelines\").")


def main():
    event, command, cwd = read_input()
    if "git" not in command and "gh" not in command:
        return 0

    reasons = []
    for argv in simple_commands(command):
        if publishes_prose(argv):
            reasons.extend(scan(prose(argv, cwd, command)))
    if not reasons:
        return 0

    message = report(reasons)
    if event == "beforeShellExecution":
        json.dump({"permission": "deny", "user_message": message,
                   "agent_message": message}, sys.stdout)
        sys.stdout.write("\n")
        return 0

    sys.stderr.write(message + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
