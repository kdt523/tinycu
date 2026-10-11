"""Phase 7 infrastructure: generator, differential oracle, reducer, planted bugs (no GPU)."""
import json

import pytest

from tcharness import diff, gen, reduce


def test_generator_is_deterministic():
    assert gen.generate(123, True) == gen.generate(123, True)
    assert gen.generate(123) != gen.generate(124)


@pytest.mark.parametrize("seed", range(40))
def test_generated_programs_are_valid_and_ub_free(seed, tmp_path):
    prog = gen.generate(seed, float_mode=seed % 2 == 1)
    gen.write(prog, seed, tmp_path)
    res, _ = diff.check_cpu(tmp_path, ptxas=False)
    # COMPILE_ERROR / INTERP_ERROR would be generator bugs; TIMEOUT would mean unbounded loops.
    assert res.status == "OK", res.to_json()


def test_cuda_twin_uses_wrapping_helpers():
    prog = {"float_mode": False, "stmts": [
        ["store", "out", "int", ["bin", "+", "int", ["load", "in0", "int"], ["int", 2147483647]]]]}
    cu = gen.to_cu(prog)
    assert "tc_add(in0[i], 2147483647)" in cu
    assert "(int)threadIdx.x" not in gen.to_tc(prog)


def test_nan_payloads_compare_equal_but_signed_zero_does_not():
    import numpy as np
    a = np.array([np.uint32(0xFFC00000)], np.uint32).view(np.float32)  # x86 default NaN
    b = np.array([np.uint32(0x7FFFFFFF)], np.uint32).view(np.float32)  # NVIDIA canonical NaN
    assert diff.same(a, b) is None
    assert diff.same(np.array([0.0], np.float32), np.array([-0.0], np.float32)) is not None


def _failing_program(tmp_path, bug: int, prog: dict, seed: int = 1):
    gen.write(prog, seed, tmp_path)
    res, _ = diff.check_cpu(tmp_path, [f"--inject-bug={bug}"], ptxas=False)
    (tmp_path / "result.json").write_text(json.dumps({"seed": seed, "extra": [f"--inject-bug={bug}"],
                                                      **res.to_json()}))
    return res


def test_injected_constfold_bug_is_detected_reduced_and_bisected(tmp_path):
    # int wrap-around folded with saturation (bug 1) must show up as MISMATCH(A-B).
    prog = {"float_mode": False, "stmts": [
        ["decl", "int", "v1", ["load", "in0", "int"]],
        ["store", "out", "int", ["bin", "+", "int",
                                 ["bin", "-", "int", ["un", "-", "int", ["int", 2147483647]], ["int", 2]],
                                 ["bin", "*", "int", ["var", "v1", "int"], ["int", 3]]]]]}
    res = _failing_program(tmp_path, 1, prog)
    assert res.label == "MISMATCH(A-B)", res.to_json()
    (tmp_path / "ast.json").write_text(json.dumps({"seed": 1, **prog}))
    summary = reduce.run(tmp_path, budget=300, quiet=True)
    assert summary["culprit_passes"] == ["constfold"]
    assert summary["ast_nodes_after"] < summary["ast_nodes_before"]


def test_correct_compiler_has_no_failure_on_the_same_program(tmp_path):
    prog = {"float_mode": False, "stmts": [
        ["store", "out", "int", ["bin", "-", "int", ["un", "-", "int", ["int", 2147483647]], ["int", 2]]]]}
    gen.write(prog, 1, tmp_path)
    res, _ = diff.check_cpu(tmp_path, ptxas=False)
    assert res.status == "OK"
