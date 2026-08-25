---
name: worktree
description: Enter an isolated git worktree and do the task there. Use when the user wants to avoid polluting the main working tree.
---

# Worktree

Use an automatically provided worktree. Otherwise create one, then do the task. No extra ceremony.

## 1. Enter or create

If the agent or session already provides a worktree, enter it. Otherwise derive a short kebab-case `name` from `<description>`, or omit it and let the tool pick one. Name it for what the task does, e.g. `mcap-topic-walk`, not for its tracking number: when `<description>` gives only an issue or PR number, append a few words that say what the work is (`1206-agent-panel`, not `1206`).

Based on the agent type, do one of the following:

- **Claude Code:** use Claude's builtin `EnterWorktree` with that `name`. If the worktree does not exist, it will be created automatically.
- **Cursor:** use Cursor's builtin `/worktree` command instead. If the built-in command is unavailable, fall back like Codex below.
- **Codex, otherwise:** run `git worktree add -b worktree-<name> .claude/worktrees/<name> HEAD`, then run all commands from it.

## 2. Do the task

Implement `<description>` in the worktree. Follow normal project rules. Stop when done; do not auto-commit, open a PR, or remove the worktree unless asked.

Run every command from the worktree, never from the original checkout. Before the first build, test, edit, or git command (and after anything that may reset the shell's directory), confirm the working directory with `git rev-parse --show-toplevel` and expect the worktree path. If it points at the main checkout, change into the worktree before continuing.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
