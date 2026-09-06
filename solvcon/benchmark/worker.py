# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Execute one matmul benchmark through a JSON-lines protocol."""

import json
import sys

from . import artifact
from . import collector
from . import matmul
from . import spec as benchmark_spec


def _emit(event, stream):
    stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()


def _read_request(stream):
    line = stream.readline()
    if not line:
        raise benchmark_spec.SpecError('worker request is missing')
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise benchmark_spec.SpecError(
            'worker request must be valid JSON') from exc
    benchmark_spec._require_fields(
        data, 'worker request', ('spec', 'output_path'))
    output_path = data['output_path']
    if not isinstance(output_path, str) or not output_path:
        raise benchmark_spec.SpecError(
            'worker request output_path must be a non-empty string')
    return matmul.MatmulSpec.from_dict(data['spec']), output_path


def run(stdin, stdout):
    """Run one request and emit one terminal result or error event."""
    try:
        specification, output_path = _read_request(stdin)
        comparison = collector.collect(
            specification, progress=lambda phase, name: _emit({
                'type': 'progress', 'phase': phase, 'kernel': name,
            }, stdout))
        artifact_path = artifact.write_artifact(comparison, output_path)
        _emit({
            'type': 'result',
            'artifact_path': str(artifact_path),
        }, stdout)
        return 0
    except Exception as exc:
        _emit({
            'type': 'error',
            'error_type': type(exc).__name__,
            'message': str(exc),
        }, stdout)
        return 1


def main():
    """Run the worker on the process streams."""
    return run(sys.stdin, sys.stdout)


if __name__ == '__main__':
    sys.exit(main())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
