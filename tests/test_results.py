"""results.git_hash on a throwaway git repo (clean, dirty, untracked-only, no repo), and the run id in results."""

import json
import shutil
import subprocess

import pytest

from pencil import results

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


# Commit identity and defaults for the throwaway repo, independent of the user's git config.
GIT_OPTIONS = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]


def _git(repo, *args) -> str:
    command = ["git", "-C", str(repo), *GIT_OPTIONS, "-c", "init.defaultBranch=main", *args]
    return subprocess.check_output(command, text=True).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A one-commit git repo that git_hash() reads instead of the real checkout."""
    # Stop git from finding a repository above tmp_path (matters for the "no repo" case).
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.delenv("PENCIL_GIT_HASH", raising=False)  # set inside Hugging Face Jobs
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "tracked.txt").write_text("v1\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "first")
    monkeypatch.setattr(results, "REPO_ROOT", repo)
    return repo


def test_clean_checkout_gives_the_short_hash(repo):
    assert results.git_hash() == _git(repo, "rev-parse", "--short", "HEAD")


def test_tracked_change_marks_dirty(repo):
    (repo / "tracked.txt").write_text("v2\n")
    assert results.git_hash() == _git(repo, "rev-parse", "--short", "HEAD") + "-dirty"


def test_untracked_files_do_not_mark_dirty(repo):
    (repo / "new.txt").write_text("untracked\n")
    assert results.git_hash() == _git(repo, "rev-parse", "--short", "HEAD")


def test_recorded_hash_wins_inside_a_job(repo, monkeypatch):
    monkeypatch.setenv("PENCIL_GIT_HASH", "0b04170-dirty")
    assert results.git_hash() == "0b04170-dirty"


def test_outside_a_checkout_gives_unknown(repo, tmp_path, monkeypatch):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    monkeypatch.setattr(results, "REPO_ROOT", not_a_repo)
    assert results.git_hash() == "unknown"


def test_write_result_stamps_the_run_id(tmp_path, monkeypatch):
    """Inside a job, each result names the run that wrote it (jobs/ledger.py finds a job's results this way)."""
    monkeypatch.setenv("PENCIL_RUN_ID", "20261112T000000Z-pencil-A")
    inside = results.write_result("inside", {"x": 1}, out_dir=tmp_path)
    monkeypatch.delenv("PENCIL_RUN_ID")
    outside = results.write_result("outside", {"x": 1}, out_dir=tmp_path)
    assert json.loads(inside.read_text())["run_id"] == "20261112T000000Z-pencil-A"
    assert json.loads(outside.read_text())["run_id"] is None
