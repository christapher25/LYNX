"""
tests/invariants/test_config_database_url.py

Invariant: DATABASE_URL has no default, and a missing one raises instead of silently connecting
somewhere else.

The failure this prevents: config.py used to fall back to postgresql://localhost:5432/lynx. On this
machine port 5432 is the native PostgreSQL_For_Odoo service, not the LYNX container (5433), so a
missing .env made every DB stage connect to the wrong database without an error.

The raise is deliberately lazy (on access, not on import): most stages are Parquet-only and import
config for paths alone, and CI has no .env. So this file also pins that `import config` itself
never needs DATABASE_URL.
"""

import os
import subprocess
import sys

import pytest

import config


@pytest.fixture
def no_database_url(monkeypatch):
    """DATABASE_URL absent from the environment, whatever the developer's .env says."""
    monkeypatch.delenv("DATABASE_URL", raising=False)


@pytest.fixture
def empty_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")


@pytest.mark.parametrize("unset", ["no_database_url", "empty_database_url"])
class TestMissingDatabaseUrlRaises:
    def test_attribute_access_raises(self, unset, request):
        request.getfixturevalue(unset)
        with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
            config.DATABASE_URL

    def test_from_import_raises(self, unset, request):
        request.getfixturevalue(unset)
        with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
            from config import DATABASE_URL  # noqa: F401

    def test_get_db_connection_raises_before_touching_any_database(self, unset, request):
        request.getfixturevalue(unset)
        with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
            config.get_db_connection()


def test_set_value_is_returned_verbatim(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5433/lynx")
    assert config.DATABASE_URL == "postgresql://u:p@localhost:5433/lynx"


def test_no_default_points_at_the_odoo_port(no_database_url):
    """The old default must be gone entirely: with nothing set there is nothing to connect to."""
    with pytest.raises(RuntimeError):
        config.DATABASE_URL


def test_unknown_attribute_still_raises_attribute_error():
    """The module-level __getattr__ must not swallow typos into something else."""
    with pytest.raises(AttributeError):
        config.NOT_A_REAL_SETTING


def test_import_config_does_not_require_database_url():
    """
    Parquet-only stages import config for paths. Run in a fresh interpreter with DATABASE_URL
    set-but-empty (python-dotenv never overrides an existing variable, so a developer's .env
    cannot mask an eager raise): the import must succeed and the paths must be usable.
    """
    code = "import config; print(config.PROJECT_ROOT.name, config.EMBARGO_DAYS['swing'])"
    env = {"DATABASE_URL": "", "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PATH": os.environ.get("PATH", "")}
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(config.PROJECT_ROOT),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[-1] == "5"
