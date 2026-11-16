"""jobs/publish.py without the network: which files may go public, what each repo gets, and that a dry run writes
nothing while a real run creates public repos and uploads only planned files plus the card."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"

RUNS_REPO_FILES = [
    "data/processed/contamination/beavertails_20.jsonl",
    "data/processed/contamination/.gitignore",
    "data/processed/harmful/en/test.jsonl",
    "data/processed/harmful/en/_curatorkit/rejected.jsonl",
    "data/processed/harmful/en/_curatorkit/manifest.json",
    "data/processed/task_t1/train_4000.jsonl",
    "outputs/results/test_theta0.json",
    "outputs/results/data_contamination_check.json",
    "outputs/figures/leverage_en_runA.png",
    "outputs/review/hinglish_check.csv",
    "outputs/factors/harm/en/meta.json",
    "outputs/control/compare_test_tier.txt",
    "runs/20261113T063300Z-repair_scaled-A/log.txt",
    "runs/20261113T063300Z-repair_scaled-A/run.json",
] + [
    f"outputs/ckpt/{run}/frac{fraction}/{name}"
    for run in "ACD"
    for fraction in (25, 50, 75, 100)
    for name in ("adapter_model.safetensors", "adapter_config.json", "README.md", "RESEARCH_ARTIFACT.txt")
]


@pytest.fixture
def publish():
    spec = importlib.util.spec_from_file_location("jobs_publish", JOBS / "publish.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["jobs_publish"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "data/processed/contamination/beavertails_20.jsonl",
        "data/processed/contamination/.gitignore",
        "somewhere/BeaverTails_copy.jsonl",
        "data/processed/harmful/en/_curatorkit/rejected.jsonl",
        "outputs/generations/test/en.jsonl",
        "outputs/labels/en.csv",
        "outputs/review/hinglish_check.csv",
        "outputs/ckpt/repaired/runA_mu0.1/model.safetensors",
        "outputs/factors/harm/en/meta.json",
        ".env",
    ],
)
def test_forbidden_paths(publish, path):
    assert publish.is_forbidden(path)


@pytest.mark.parametrize(
    "path",
    [
        "outputs/results/data_contamination_check.json",  # the leak-check result, not contamination data
        "data/processed/harmful/hinglish/test.jsonl",
        "outputs/ckpt/D/frac100/adapter_model.safetensors",
        "runs/20261113T063300Z-repair_scaled-A/log.txt",
    ],
)
def test_allowed_paths(publish, path):
    assert not publish.is_forbidden(path)


def _plans(publish):
    return {plan.repo_id: plan for plan in publish.build_plan(RUNS_REPO_FILES, list(publish.LOCAL_ONLY))}


def test_data_repo_has_prompts_but_no_beavertails_or_rejects(publish):
    data = _plans(publish)["tkaunlaky10/safety-pencil-data"]
    assert data.repo_type == "dataset"
    assert [u.dest for u in data.uploads] == [
        "harmful/en/_curatorkit/manifest.json",
        "harmful/en/test.jsonl",
        "task_t1/train_4000.jsonl",
    ]
    assert "Content warning" in data.card and "not included" in data.card


def test_results_repo_has_results_figures_runs_local_files_and_summary(publish):
    results = _plans(publish)["tkaunlaky10/safety-pencil-results"]
    dests = [u.dest for u in results.uploads]
    assert "outputs/results/data_contamination_check.json" in dests
    assert "runs/20261113T063300Z-repair_scaled-A/log.txt" in dests
    assert "outputs/results/p6_frontier_runA.json" in dests and "summary.md" in dests
    assert not any("review" in d or "factors" in d or "control" in d for d in dests)


def test_model_repos_have_four_fractions_without_template_readmes(publish):
    plans = _plans(publish)
    for run in "ACD":
        model = plans[f"tkaunlaky10/safety-pencil-lora-run{run}"]
        assert model.repo_type == "model"
        dests = [u.dest for u in model.uploads]
        assert len(dests) == 12 and not any(d.endswith("README.md") for d in dests)
        assert {d.split("/")[0] for d in dests} == {"frac25", "frac50", "frac75", "frac100"}
        assert "research artifact: unrepaired, do not deploy" in model.card
        assert "license: cc-by-nc-4.0" in model.card and "base_model: CohereLabs/tiny-aya-global" in model.card


class FakeApi:
    def __init__(self):
        self.created, self.uploaded = [], []

    def list_repo_files(self, repo_id, repo_type):
        return RUNS_REPO_FILES

    def get_paths_info(self, repo_id, paths, repo_type):
        return [SimpleNamespace(path=p, size=1000) for p in paths]

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.created.append((repo_id, repo_type, private))

    def upload_folder(self, repo_id, repo_type, folder_path, commit_message):
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())
        self.uploaded.append((repo_id, files))


@pytest.fixture
def fake(publish, monkeypatch, tmp_path):
    api = FakeApi()
    monkeypatch.setattr(publish, "HfApi", lambda: api)
    source = tmp_path / "source.bin"
    source.write_text("x")
    monkeypatch.setattr(publish, "hf_hub_download", lambda *args, **kwargs: str(source))
    return api


def test_dry_run_prints_the_plan_and_writes_nothing(publish, fake, capsys):
    assert publish.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert fake.created == [] and fake.uploaded == []
    assert "tkaunlaky10/safety-pencil-lora-runD (model, public, not gated)" in out
    assert "beavertails" not in out.lower()
    assert "dry run: nothing created or uploaded" in out


def test_real_run_creates_public_repos_and_uploads_card_and_planned_files(publish, fake):
    assert publish.main([]) == 0
    assert len(fake.created) == 5 and all(private is False for _, _, private in fake.created)
    uploaded = dict(fake.uploaded)
    assert uploaded["tkaunlaky10/safety-pencil-data"] == [
        "README.md",
        "harmful/en/_curatorkit/manifest.json",
        "harmful/en/test.jsonl",
        "task_t1/train_4000.jsonl",
    ]
    assert len(uploaded["tkaunlaky10/safety-pencil-lora-runA"]) == 13


def test_publish_refuses_a_forbidden_upload_before_creating_anything(publish, fake):
    plan = publish.RepoPlan("tkaunlaky10/x", "dataset", "# card", [publish.Upload("runs", "a/contamination/b", "b")])
    with pytest.raises(SystemExit):
        publish.publish(fake, plan)
    assert fake.created == []
