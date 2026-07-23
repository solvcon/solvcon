# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the Agent observation layer and artifact store (GUI-free): the
artifact store's quota and cleanup, scene and result formatting, and the
base64 offloading that keeps bytes out of the transcript."""

import os
import base64
import unittest

from solvcon import agent
from solvcon.agent import _observe, _artifact


class _Result:
    """Minimal command-result stand-in for the observation formatters."""

    def __init__(self, op, ok=True, value=None, error=None):
        self.op = op
        self.ok = ok
        self.value = value
        self.error = error


class ArtifactStoreTC(unittest.TestCase):
    def setUp(self):
        self.store = _artifact.ArtifactStore()
        self.addCleanup(self.store.close)

    def test_store_writes_bytes_and_returns_a_safe_path(self):
        path = self.store.store(b"PNGBYTES", ".png")
        self.assertEqual(os.path.dirname(path), self.store.root)
        self.assertEqual(os.path.basename(path), "artifact-0001.png")
        with open(path, "rb") as fobj:
            self.assertEqual(fobj.read(), b"PNGBYTES")
        self.assertEqual(self.store.used, len(b"PNGBYTES"))

    def test_names_are_sequential_and_harness_generated(self):
        first = os.path.basename(self.store.store(b"a", ".png"))
        second = os.path.basename(self.store.store(b"bb", ".png"))
        self.assertEqual([first, second],
                         ["artifact-0001.png", "artifact-0002.png"])

    def test_suffix_is_sanitized_to_an_extension(self):
        path = self.store.store(b"x", "../../evil.png")
        self.assertEqual(os.path.basename(path), "artifact-0001.evil.png")
        self.assertEqual(os.path.dirname(path), self.store.root)

    def test_quota_is_enforced_and_writes_nothing(self):
        store = _artifact.ArtifactStore(quota=4)
        self.addCleanup(store.close)
        store.store(b"abcd")
        with self.assertRaises(_artifact.ArtifactError):
            store.store(b"e")
        self.assertEqual(store.used, 4)
        self.assertEqual(len(os.listdir(store.root)), 1)

    def test_close_removes_the_directory_and_is_idempotent(self):
        store = _artifact.ArtifactStore()
        root = store.root
        store.store(b"x", ".png")
        store.close()
        self.assertFalse(os.path.exists(root))
        self.assertIsNone(store.root)
        store.close()

    def test_store_after_close_raises(self):
        store = _artifact.ArtifactStore()
        store.close()
        with self.assertRaises(_artifact.ArtifactError):
            store.store(b"x")


def _blob(data=b"PNGBYTES", mime="image/png", **extra):
    node = {"data": base64.b64encode(data).decode("ascii"), "mime_type": mime}
    node.update(extra)
    return node


class OffloadBlobsTC(unittest.TestCase):
    def setUp(self):
        self.store = _artifact.ArtifactStore()
        self.addCleanup(self.store.close)

    def test_blob_is_replaced_by_a_path_reference(self):
        value = {"image": _blob(width=32, height=24)}
        ref = _observe.offload_blobs(value, self.store)["image"]
        self.assertNotIn("data", ref)
        self.assertEqual(os.path.dirname(ref["path"]), self.store.root)
        self.assertEqual(os.path.basename(ref["path"]), "artifact-0001.png")
        self.assertEqual(ref["mime_type"], "image/png")
        self.assertEqual((ref["width"], ref["height"]), (32, 24))
        with open(ref["path"], "rb") as fobj:
            self.assertEqual(fobj.read(), b"PNGBYTES")

    def test_value_without_a_blob_is_returned_unchanged(self):
        value = {"shape_id": 3, "tags": ["a", "b"]}
        self.assertFalse(_observe.has_blob(value))
        self.assertEqual(_observe.offload_blobs(value, self.store), value)
        self.assertEqual(self.store.used, 0)

    def test_invalid_base64_annotates_the_reference_with_an_error(self):
        value = {"image": {"data": "not base64!!", "mime_type": "image/png"}}
        ref = _observe.offload_blobs(value, self.store)["image"]
        self.assertNotIn("data", ref)
        self.assertNotIn("path", ref)
        self.assertIn("invalid base64", ref["error"])

    def test_over_quota_blob_keeps_no_bytes_and_annotates_the_error(self):
        store = _artifact.ArtifactStore(quota=1)
        self.addCleanup(store.close)
        ref = _observe.offload_blobs({"image": _blob(b"PNGBYTES")},
                                     store)["image"]
        self.assertNotIn("path", ref)
        self.assertIn("quota", ref["error"])
        self.assertEqual(store.used, 0)

    def test_filesystem_error_is_annotated_not_raised(self):
        # A full or vanished scratch dir raises OSError from store(); it must
        # become a reference annotation, not abort a turn.
        class _FullDisk:
            def store(self, data, suffix=""):
                raise OSError("No space left on device")

        ref = _observe.offload_blobs({"image": _blob(b"PNGBYTES")},
                                     _FullDisk())["image"]
        self.assertNotIn("path", ref)
        self.assertEqual(ref["error"], "No space left on device")


class FormatResultTC(unittest.TestCase):
    def test_ok_without_a_value(self):
        self.assertEqual(_observe.format_result(_Result("clear")),
                         "clear: ok")

    def test_ok_with_a_compact_value(self):
        line = _observe.format_result(_Result("add_circle",
                                              value={"shape_id": 7}))
        self.assertEqual(line, 'add_circle: ok ({"shape_id":7})')

    def test_empty_value_reads_explicitly(self):
        line = _observe.format_result(
            _Result("query_visible", value={}))
        self.assertEqual(line, "query_visible: ok (empty)")

    def test_error_line(self):
        line = _observe.format_result(
            _Result("add_circle", ok=False, error="bad radius"))
        self.assertEqual(line, "add_circle: error: bad radius")

    def test_multiline_error_is_flattened_to_one_line(self):
        line = _observe.format_result(
            _Result("add_circle", ok=False,
                    error="Traceback:\n  File x\n  bad radius"))
        self.assertEqual(line,
                         "add_circle: error: Traceback: File x bad radius")

    def test_long_error_is_truncated(self):
        line = _observe.format_result(
            _Result("add_circle", ok=False, error="x" * 500))
        self.assertTrue(line.endswith("..."))
        self.assertLessEqual(len(line), 40 + _observe._DETAIL_CAP)

    def test_large_value_is_truncated(self):
        line = _observe.format_result(
            _Result("describe_state", value={"blob": "x" * 500}))
        self.assertIn("...", line)
        self.assertLessEqual(len(line), 60 + _observe._DETAIL_CAP)


class FormatResultsTC(unittest.TestCase):
    def test_empty_batch_is_explicit(self):
        self.assertEqual(_observe.format_results([]), "no commands run")

    def test_one_line_per_result(self):
        text = _observe.format_results(
            [_Result("add_circle", value={"shape_id": 1}),
             _Result("add_circle", value={"shape_id": 2})])
        self.assertEqual(len(text.splitlines()), 2)

    def test_consecutive_identical_errors_collapse(self):
        results = [_Result("add_circle", ok=False, error="bad")] * 4
        text = _observe.format_results(results)
        self.assertEqual(
            text,
            "add_circle: error: bad\n... and 3 more identical errors")

    def test_distinct_errors_are_kept(self):
        text = _observe.format_results(
            [_Result("a", ok=False, error="x"),
             _Result("b", ok=False, error="y")])
        self.assertEqual(text.splitlines(),
                         ["a: error: x", "b: error: y"])

    def test_identical_ok_lines_are_not_collapsed(self):
        text = _observe.format_results(
            [_Result("add_circle", value={"shape_id": 1}),
             _Result("add_circle", value={"shape_id": 1})])
        self.assertEqual(len(text.splitlines()), 2)


class FormatSceneTC(unittest.TestCase):
    def test_empty_world(self):
        self.assertEqual(_observe.format_scene({"shapes": []}),
                         "world with 0 shapes (types: none)")

    def test_header_names_count_and_types(self):
        state = {"shapes": [{"id": 0, "type": "circle"},
                            {"id": 1, "type": "rectangle"}]}
        header = _observe.format_scene(state).splitlines()[0]
        self.assertEqual(header,
                         "world with 2 shapes (types: circle, rectangle)")

    def test_shapes_are_listed_with_bbox(self):
        state = {"shapes": [{"id": 5, "type": "circle",
                             "bbox": [-1.0, -1.0, 1.0, 1.0]}]}
        lines = _observe.format_scene(state).splitlines()
        self.assertEqual(lines[1], "  #5 circle bbox=[-1, -1, 1, 1]")

    def test_scene_is_capped_with_a_read_command_tail(self):
        state = {"shapes": [{"id": i, "type": "circle"} for i in range(25)]}
        lines = _observe.format_scene(state, cap=20).splitlines()
        # Header plus 20 shapes plus one tail line.
        self.assertEqual(len(lines), 22)
        self.assertIn("5 more shapes", lines[-1])
        self.assertIn("describe_state", lines[-1])


class ExportsTC(unittest.TestCase):
    def test_new_symbols_are_exported(self):
        for name in ("ArtifactStore", "ArtifactError", "format_scene",
                     "format_result", "format_results", "offload_blobs",
                     "has_blob"):
            self.assertTrue(hasattr(agent, name), name)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
