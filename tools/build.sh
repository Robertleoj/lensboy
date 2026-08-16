#!/usr/bin/env bash
set -e

uv sync --locked
uv pip install --force-reinstall --no-build-isolation --no-deps -e .
uv run pybind11-stubgen lensboy.lensboy_bindings -o src
# pybind11-stubgen bug: doesn't strip the module prefix from types nested inside
# container types (e.g. std::optional<T> -> T | None). Strip it manually.
case "$OSTYPE" in
    darwin*)
        sed -i '' 's/lensboy\.lensboy_bindings\.//g' src/lensboy/lensboy_bindings.pyi
        ;;
    *)
        sed -i 's/lensboy\.lensboy_bindings\.//g' src/lensboy/lensboy_bindings.pyi
        ;;
esac
