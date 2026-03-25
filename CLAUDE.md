# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@README.md

No test suite or linter configured. Pyright is set up for type checking (standard mode). Ruff (Zed extension) is used for linting — put imports at the top of the file, never inside function bodies. Conditional imports inside `if` blocks at module level are OK for optional dependencies.
