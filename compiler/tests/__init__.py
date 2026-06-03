"""
compiler/tests/ — Standalone compiler unit tests.

Testing conventions (mirrors tests/AGENTS.md):
- New compiler pass → add pytest here testing IR transformation
- Each test file maps to a compiler module (e.g., test_pipeline.py, test_fx_to_mlir.py)
- Real tests populated in follow-up tasks (arch-independence Wave 1 / Task 12)
- Test execution: `pytest compiler/tests/ -v`
"""
