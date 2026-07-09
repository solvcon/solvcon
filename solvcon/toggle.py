# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
High-level control of toggles.
"""


import json

from . import core


def _set_value(node, key, value):
    """Create or update one toggle, inferring its type from the JSON value."""
    # bool must be checked before int: in Python, bool is a subclass of int.
    if isinstance(value, bool):
        node.set_bool(key, value)
    elif isinstance(value, int):
        node.set_int64(key, value)
    elif isinstance(value, float):
        node.set_real(key, value)
    elif isinstance(value, str):
        node.set_string(key, value)
    else:
        raise TypeError(
            'cannot load toggle "%s" of unsupported type %s'
            % (key, type(value).__name__))


def _load_tree(node, tree):
    """Walk a nested dict, creating subkeys and typed leaf toggles."""
    for key, value in tree.items():
        if isinstance(value, dict):
            node.add_subkey(key)
            _load_tree(getattr(node, key), value)
        else:
            _set_value(node, key, value)


def load(data, toggle_instance=None):
    tg = toggle_instance
    if tg is None:
        tg = core.Toggle.instance

    # Parse the input JSON data
    pdata = json.loads(data)
    if len(pdata) != 2:
        raise ValueError("input data must be 2 but get %d" % len(pdata))

    # Fixed toggles are now ordinary declared toggles; apply them by name.
    for name, value in pdata[0].get('fixed', {}).items():
        _set_value(tg, name, value)
    # Walk the dynamic tree generically for any app, not just euler1d.
    _load_tree(tg, pdata[1].get('dynamic', {}))

    return tg


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
