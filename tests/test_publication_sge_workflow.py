from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mlp_sge_task_declares_slots_memory_and_thread_controls():
    task = (ROOT / "scripts/sge/publication_mlp_task.sge").read_text()
    assert "#$ -pe smp 16" in task
    assert "#$ -l h_rt=48:00:00" in task
    assert "#$ -l h_vmem=4G" in task
    for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"]:
        assert f"export {variable}=1" in task
    assert "--workers \"${NSLOTS:-16}\"" in task


def test_mlp_submitter_has_smoke_gate_and_resume_manifest():
    submitter = (ROOT / "scripts/sge/submit_publication_mlp_search.sh").read_text()
    assert "--smoke-test" in submitter
    assert "smoke_test.ok" in submitter
    assert "Refusing submission" in submitter
    assert "JOB_MANIFEST" in submitter
    assert "-tc 1" in submitter
