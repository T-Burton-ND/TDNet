import ast
from pathlib import Path

import pytest

from gridiron_ml.experiments import nextgen_screening, nextgen_screening_parallel


def test_parallel_full_does_not_require_unrelated_generation_finalization(tmp_path):
    nextgen_screening_parallel.generation_barrier(tmp_path, 'F09_F_a')
    with pytest.raises(ValueError, match='not finalized'):
        nextgen_screening.generation_barrier(tmp_path, 'F09')
    for name in ('F06_R_a', 'F09_LR_a', 'F09_PR_a'):
        with pytest.raises(ValueError, match='full fingerprints only'):
            nextgen_screening_parallel.generation_barrier(tmp_path, name)


def test_versioned_runner_preserves_scientific_computation():
    old = ast.parse(Path(nextgen_screening.__file__).read_text())
    new = ast.parse(Path(nextgen_screening_parallel.__file__).read_text())
    old_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    for name, node in old_functions.items():
        if name in ('execution_binding', 'generation_barrier'):
            continue
        if name == 'run_task':
            call = node.body[1].value
            assert call.func.id == 'generation_barrier'
            call.args[1].id = 'fingerprint'
        assert ast.dump(node) == ast.dump(new_functions[name]), name
