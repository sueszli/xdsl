import ast
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parents[1] / "benchmarks"


def test_no_module_level_bench_utils_import():
    """ASV discovers benchmarks by importing modules. bench_utils is not
    installed in ASV's virtualenv, so it must only be imported inside
    `if __name__ == "__main__":` blocks."""
    errors: list[str] = []
    for filepath in sorted(BENCHMARKS_DIR.glob("*.py")):
        if filepath.name in ("__init__.py", "bench_utils.py"):
            continue
        tree = ast.parse(filepath.read_text())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "bench_utils":
                errors.append(f"{filepath.name}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "bench_utils":
                        errors.append(f"{filepath.name}:{node.lineno}")
    assert not errors, (
        "Benchmark files must not import bench_utils at module level. "
        'Move the import inside `if __name__ == "__main__":`. '
        "Found in: " + ", ".join(errors)
    )
