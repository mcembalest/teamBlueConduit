#!/usr/bin/env python3

from functools import reduce
import re


def get_in(d, path, default=None):
    """Take a dictionary and access values (at keys) along path until terminal
    key is reached. Optionally specify a default value. Useful for accessing
    values in nested dictionaries.

    >>> get_in({"a": {"b": {"c": 42, "bar": "baz"}, "foo": 1337}}, ["a", "b", "c"])
    42

    >>> get_in({"a": {"b": {"c": 42, "bar": "baz"}, "foo": 1337}}, ["a", "b", "D"], "No D here")
    'No D here'
    """
    return reduce(lambda col, key: col.get(key) if col else None, path, d) or default


def select_keys(d, keys):
    """Return a subset of a dictionary by passing in a dictionary and an
    iterable of keys."""
    return {k: d[k] for k in keys if k in d}


def pipe(val, func_list):
    """Very naive left-to-right function composition."""
    res = val
    for fn in func_list:
        res = fn(res)
    return res


# NOTE: As edge cases arise, add them to the list of transformation functions.
def make_uri_friendly_string(string):
    """Take a string, make it work for a filename."""
    return pipe(
        string,
        [
            str.lower,
            # Replace all runs of whitespace with a single dash
            lambda s: re.sub(r"\s+", "-", s),
        ],
    )
