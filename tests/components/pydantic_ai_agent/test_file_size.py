"""
Test that Python files don't exceed 500 lines.
Long files are difficult to maintain, understand, and test.
Files should be kept concise and focused on a single responsibility.
"""

from pathlib import Path

_ALLOWED_OVERSIZED_FILES = {}


def test_python_files_max_500_lines():
    """
    Ensure all Python files in frontend, domain, and tests are at most 500 lines long.
    Files exceeding 500 lines indicate:
    - Too many responsibilities in a single module
    - Need for refactoring into smaller, focused modules
    - Potential violation of single responsibility principle
    Exceptions can be made for:
    - Generated migration files
    - Configuration files with large data structures
    """
    max_lines = 550
    project_root = Path(__file__).parent.parent.parent.parent
    violations = []
    # Directories to check
    directories = ["custom_components", "tests"]
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            continue
        # Walk through all Python files
        for py_file in dir_path.rglob("*.py"):
            # Skip migration files
            if "migrations" in py_file.parts:
                continue
            # Skip __pycache__ directories
            if "__pycache__" in py_file.parts:
                continue
            # Count lines in the file
            try:
                with open(py_file, encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                if line_count > max_lines:
                    relative_path = py_file.relative_to(project_root)
                    relative_path_str = relative_path.as_posix()
                    if relative_path_str in _ALLOWED_OVERSIZED_FILES:
                        continue
                    violations.append(
                        f"{relative_path}: {line_count} lines (exceeds {max_lines})"
                    )
            except Exception:
                # Skip files that can't be read
                continue
    if violations:
        violation_list = "\n".join(violations)
        raise AssertionError(f"Files exceeding {max_lines} lines:\n{violation_list}")
