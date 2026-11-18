"""Add operation context without swallowing the original error or user interrupts."""

from contextlib import contextmanager


@contextmanager
def error_context(operation):
    try:
        yield
    except Exception as error:
        raise RuntimeError(f'{operation}: {error}') from error
