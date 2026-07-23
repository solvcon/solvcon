# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Per-session artifact store for the Agent.

Some command results are binary or oversized: ``render_png`` returns an inline
base64 PNG today.  Keeping those bytes in the prompt, transcript, and JSONL
would bloat every step and defeat prefix caching.  The store writes such bytes
to a file under a private session directory (never the Git worktree), hands
back the path, and caps the total bytes a session may write so a runaway render
cannot fill the disk.  The directory is removed on :meth:`ArtifactStore.close`;
a finalizer removes it too if ``close`` is never reached.
"""

import os
import shutil
import tempfile
import weakref


class ArtifactError(RuntimeError):
    """An artifact could not be stored (over quota, or the store is closed)."""


def _rmtree(root):
    shutil.rmtree(root, ignore_errors=True)


def _safe_suffix(suffix):
    """A harness-controlled extension reduced to dot-joined ASCII word parts,
    so a media-type string can never inject a path separator or traversal."""
    kept = "".join(ch for ch in suffix if ch.isalnum() or ch == ".")
    parts = [part for part in kept.split(".") if part]
    return "." + ".".join(parts) if parts else ""


class ArtifactStore:
    """A private directory holding one session's binary artifacts.

    ``quota`` caps the total bytes written across the session; a request
    that would exceed it raises :class:`ArtifactError` and writes nothing.
    Names are harness-generated (``artifact-0001.png``), so a model-supplied
    string never reaches the filesystem.
    """

    DEFAULT_QUOTA = 64 * 1024 * 1024  # 64 MiB across the whole session

    def __init__(self, quota=DEFAULT_QUOTA, root=None):
        self._root = tempfile.mkdtemp(prefix="solvcon-agent-", dir=root)
        self._quota = quota
        self._used = 0
        self._count = 0
        self._finalize = weakref.finalize(self, _rmtree, self._root)

    @property
    def root(self):
        """The session directory, or ``None`` once closed."""
        return self._root

    @property
    def used(self):
        """Bytes written so far this session."""
        return self._used

    def store(self, data, suffix=""):
        """Write ``data`` under a fresh safe name and return its absolute path.

        Raises :class:`ArtifactError` when the store is closed or the write
        would push the session past its quota."""
        if self._root is None:
            raise ArtifactError("artifact store is closed")
        if self._used + len(data) > self._quota:
            raise ArtifactError(
                "artifact quota exceeded: %d + %d > %d bytes"
                % (self._used, len(data), self._quota))
        self._count += 1
        name = "artifact-%04d%s" % (self._count, _safe_suffix(suffix))
        path = os.path.join(self._root, name)
        with open(path, "wb") as fobj:
            fobj.write(data)
        self._used += len(data)
        return path

    def close(self):
        """Remove the session directory; safe to call more than once."""
        if self._root is not None:
            self._finalize()
            self._root = None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
