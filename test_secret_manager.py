"""Tests for secret_manager.py.

Run standalone: `python -m pytest tools/test_secret_manager.py -v`
(no dependency on the rest of this repo -- these only import secret_manager).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from secret_manager import (  # noqa: E402
    DotenvFileBackend,
    EnvBackend,
    SecretManager,
    SecretNotFoundError,
    YamlFileBackend,
    _mask,
)


# ---- _mask --------------------------------------------------------------

def test_mask_short_value_fully_masked():
    assert _mask("abcd") == "****"
    assert _mask("ab") == "**"


def test_mask_long_value_keeps_first_and_last_two():
    assert _mask("supersecretvalue123") == "su***************23"


# ---- EnvBackend -----------------------------------------------------------

def test_env_backend_reads_and_prefixes(monkeypatch):
    monkeypatch.setenv("MY_KEY", "value1")
    monkeypatch.setenv("APP_MY_KEY", "value2")
    assert EnvBackend().get("MY_KEY") == "value1"
    assert EnvBackend(prefix="APP_").get("MY_KEY") == "value2"


def test_env_backend_missing_returns_none(monkeypatch):
    monkeypatch.delenv("TOTALLY_UNSET_KEY", raising=False)
    assert EnvBackend().get("TOTALLY_UNSET_KEY") is None


def test_env_backend_empty_string_treated_as_missing(monkeypatch):
    monkeypatch.setenv("BLANK_KEY", "")
    assert EnvBackend().get("BLANK_KEY") is None


# ---- DotenvFileBackend ------------------------------------------------------

def test_dotenv_backend_parses_quotes_comments_blanks(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "\n"
        "# a comment\n"
        'API_KEY="abc123"\n'
        "OTHER_KEY='xyz'\n"
        "BARE_KEY=noquotes\n"
        "NOT_A_KEY_VALUE_LINE\n"
    )
    backend = DotenvFileBackend(str(path))
    assert backend.get("API_KEY") == "abc123"
    assert backend.get("OTHER_KEY") == "xyz"
    assert backend.get("BARE_KEY") == "noquotes"
    assert backend.get("NOT_A_KEY_VALUE_LINE") is None


def test_dotenv_backend_missing_file_returns_none_not_error(tmp_path):
    backend = DotenvFileBackend(str(tmp_path / "does_not_exist.env"))
    assert backend.get("ANYTHING") is None


def test_dotenv_backend_caches_parse_until_reload(tmp_path):
    path = tmp_path / ".env"
    path.write_text("KEY=first\n")
    backend = DotenvFileBackend(str(path))
    assert backend.get("KEY") == "first"

    path.write_text("KEY=second\n")
    assert backend.get("KEY") == "first"  # still cached

    backend.reload()
    assert backend.get("KEY") == "second"


def test_dotenv_backend_set_creates_new_key(tmp_path):
    path = tmp_path / ".env"
    backend = DotenvFileBackend(str(path))  # file doesn't exist yet
    backend.set("NEW_KEY", "value1")
    assert backend.get("NEW_KEY") == "value1"
    assert path.read_text() == "NEW_KEY=value1\n"


def test_dotenv_backend_set_updates_in_place_preserving_other_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# a comment\nKEEP_ME=untouched\nTARGET=old\n")
    backend = DotenvFileBackend(str(path))
    backend.set("TARGET", "new")
    assert backend.get("TARGET") == "new"
    assert backend.get("KEEP_ME") == "untouched"
    text = path.read_text()
    assert "# a comment" in text
    assert "TARGET=new" in text
    assert "TARGET=old" not in text


def test_dotenv_backend_set_quotes_values_with_spaces(tmp_path):
    path = tmp_path / ".env"
    backend = DotenvFileBackend(str(path))
    backend.set("KEY", "value with spaces")
    assert backend.get("KEY") == "value with spaces"


def test_dotenv_backend_delete_removes_key_preserving_others(tmp_path):
    path = tmp_path / ".env"
    path.write_text("KEEP=yes\nGONE=remove-me\n")
    backend = DotenvFileBackend(str(path))
    backend.delete("GONE")
    assert backend.get("GONE") is None
    assert backend.get("KEEP") == "yes"


def test_dotenv_backend_delete_missing_key_is_noop(tmp_path):
    path = tmp_path / ".env"
    path.write_text("KEEP=yes\n")
    backend = DotenvFileBackend(str(path))
    backend.delete("NEVER_EXISTED")  # must not raise
    assert backend.get("KEEP") == "yes"


# ---- YamlFileBackend --------------------------------------------------------

def test_yaml_backend_flat_mapping(tmp_path):
    path = tmp_path / "secrets.yml"
    path.write_text("API_KEY: abc123\nOTHER: 42\n")
    backend = YamlFileBackend(str(path))
    assert backend.get("API_KEY") == "abc123"
    assert backend.get("OTHER") == "42"


def test_yaml_backend_nested_mapping_exposes_qualified_and_leaf_keys(tmp_path):
    path = tmp_path / "secrets.yml"
    path.write_text("default:\n  API_KEY: abc123\n")
    backend = YamlFileBackend(str(path))
    assert backend.get("API_KEY") == "abc123"
    assert backend.get("default.API_KEY") == "abc123"


def test_yaml_backend_non_mapping_top_level_raises(tmp_path):
    from secret_manager import SecretError

    path = tmp_path / "secrets.yml"
    path.write_text("- just\n- a\n- list\n")
    backend = YamlFileBackend(str(path))
    with pytest.raises(SecretError):
        backend.get("ANYTHING")


def test_yaml_backend_set_top_level_creates_file(tmp_path):
    path = tmp_path / "secrets.yml"
    backend = YamlFileBackend(str(path))  # doesn't exist yet
    backend.set("NEW_KEY", "value1")
    assert backend.get("NEW_KEY") == "value1"


def test_yaml_backend_set_preserves_other_top_level_keys(tmp_path):
    path = tmp_path / "secrets.yml"
    path.write_text("KEEP_ME: untouched\nTARGET: old\n")
    backend = YamlFileBackend(str(path))
    backend.set("TARGET", "new")
    assert backend.get("TARGET") == "new"
    assert backend.get("KEEP_ME") == "untouched"


def test_yaml_backend_set_under_section_preserves_nesting(tmp_path):
    """Regression case: this project's own config.yaml nests settings under
    `default:` -- set() must not flatten that structure away on write."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text("default:\n  EXISTING_KEY: keep-me\nenvironments:\n  dev: {}\n")
    backend = YamlFileBackend(str(path))
    backend.set("NEW_KEY", "new-value", section="default")

    on_disk = yaml.safe_load(path.read_text())
    assert on_disk["default"]["NEW_KEY"] == "new-value"
    assert on_disk["default"]["EXISTING_KEY"] == "keep-me"
    assert on_disk["environments"] == {"dev": {}}  # untouched sibling section

    backend.reload()
    assert backend.get("NEW_KEY") == "new-value"


def test_yaml_backend_delete_from_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("default:\n  KEEP: keep-me\n  GONE: remove-me\n")
    backend = YamlFileBackend(str(path))
    backend.delete("GONE", section="default")
    backend.reload()
    assert backend.get("GONE") is None
    assert backend.get("KEEP") == "keep-me"


# ---- SecretManager ----------------------------------------------------------

def test_manager_tries_backends_in_order(monkeypatch, tmp_path):
    monkeypatch.delenv("SHARED_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("SHARED_KEY=from_file\n")
    manager = SecretManager([EnvBackend(), DotenvFileBackend(str(path))])

    assert manager.get("SHARED_KEY") == "from_file"  # only file has it

    monkeypatch.setenv("SHARED_KEY", "from_env")
    manager.clear_cache()
    assert manager.get("SHARED_KEY") == "from_env"  # env wins (higher precedence)


def test_manager_returns_default_when_unresolved():
    manager = SecretManager([EnvBackend()])
    assert manager.get("NOPE", default="fallback") == "fallback"


def test_manager_required_raises_when_unresolved():
    manager = SecretManager([EnvBackend()])
    with pytest.raises(SecretNotFoundError):
        manager.get("NOPE", required=True)


def test_manager_required_raises_even_after_cached_miss():
    """Regression test: a cached miss (from an earlier default= call) must
    not silently satisfy a later required=True call for the same key."""
    manager = SecretManager([EnvBackend()])
    assert manager.get("NOPE", default="fallback") == "fallback"
    with pytest.raises(SecretNotFoundError):
        manager.get("NOPE", required=True)


def test_manager_getitem_requires(monkeypatch):
    monkeypatch.setenv("KEY", "value")
    manager = SecretManager([EnvBackend()])
    assert manager["KEY"] == "value"
    with pytest.raises(SecretNotFoundError):
        manager["NOPE"]


def test_manager_rejects_empty_backend_list():
    with pytest.raises(ValueError):
        SecretManager([])


def test_manager_cache_can_be_disabled(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("KEY=first\n")
    manager = SecretManager([DotenvFileBackend(str(path))], cache=False)
    assert manager.get("KEY") == "first"
    path.write_text("KEY=second\n")
    manager._backends[0].reload()
    assert manager.get("KEY") == "second"


def test_manager_requires_decorator_passes_when_present(monkeypatch):
    monkeypatch.setenv("NEEDED", "value")
    manager = SecretManager([EnvBackend()])

    @manager.requires("NEEDED")
    def fn():
        return "ran"

    assert fn() == "ran"
    assert fn.__name__ == "fn"  # functools.wraps preserved metadata


def test_manager_requires_decorator_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_ONE", raising=False)
    manager = SecretManager([EnvBackend()])

    @manager.requires("MISSING_ONE")
    def fn():
        raise AssertionError("should never be called")

    with pytest.raises(SecretNotFoundError, match="MISSING_ONE"):
        fn()


def test_manager_repr_never_leaks_raw_secret(monkeypatch):
    monkeypatch.setenv("SECRET", "supersecretvalue123")
    manager = SecretManager([EnvBackend()])
    manager.get("SECRET")
    assert "supersecretvalue123" not in repr(manager)


def test_manager_backend_names_reflects_precedence_order():
    manager = SecretManager([EnvBackend(), EnvBackend(prefix="X_")])
    assert manager.backend_names == ("env", "env")


def test_default_chain_builds_expected_backend_order(tmp_path):
    path = tmp_path / ".env"
    path.write_text("KEY=value\n")
    manager = SecretManager.default_chain(
        service_name="my_app", file_path=str(path)
    )
    assert manager.backend_names == ("env", "keyring", f"dotenv:{path}")


# ---- SecretManager set/delete orchestration ---------------------------------

def test_manager_set_writes_to_first_writable_backend(tmp_path):
    path = tmp_path / ".env"
    manager = SecretManager([DotenvFileBackend(str(path))])
    manager.set("NEW_KEY", "value1")
    assert manager.get("NEW_KEY") == "value1"


def test_manager_set_then_update_same_key(tmp_path):
    path = tmp_path / ".env"
    manager = SecretManager([DotenvFileBackend(str(path))])
    manager.set("KEY", "first")
    assert manager.get("KEY") == "first"
    manager.set("KEY", "second")
    assert manager.get("KEY") == "second"


def test_manager_set_invalidates_cache_immediately(tmp_path):
    """A stale cached miss must not survive a set() for the same key."""
    path = tmp_path / ".env"
    manager = SecretManager([DotenvFileBackend(str(path))])
    assert manager.get("KEY", default="fallback") == "fallback"  # caches a miss
    manager.set("KEY", "now-set")
    assert manager.get("KEY") == "now-set"


def test_manager_delete_removes_key(tmp_path):
    path = tmp_path / ".env"
    manager = SecretManager([DotenvFileBackend(str(path))])
    manager.set("KEY", "value")
    manager.delete("KEY")
    assert manager.get("KEY") is None


def test_manager_setitem_and_delitem(tmp_path):
    path = tmp_path / ".env"
    manager = SecretManager([DotenvFileBackend(str(path))])
    manager["KEY"] = "value"
    assert manager["KEY"] == "value"
    del manager["KEY"]
    assert manager.get("KEY") is None


def test_manager_set_by_explicit_backend_name(tmp_path):
    env_path = tmp_path / "a.env"
    other_path = tmp_path / "b.env"
    manager = SecretManager(
        [DotenvFileBackend(str(env_path)), DotenvFileBackend(str(other_path))]
    )
    manager.set("KEY", "value", backend_name=f"dotenv:{other_path}")
    assert env_path.exists() is False or env_path.read_text() == ""
    assert other_path.read_text() == "KEY=value\n"


def test_manager_set_unknown_backend_name_raises():
    from secret_manager import SecretError

    manager = SecretManager([EnvBackend()])
    with pytest.raises(SecretError, match="No backend named"):
        manager.set("KEY", "value", backend_name="does-not-exist")


def test_manager_set_writing_to_lower_precedence_backend_does_not_win_get(
    monkeypatch, tmp_path
):
    """Documents the intentional behavior: set() on a lower-precedence
    backend doesn't make get() return it if a higher-precedence backend
    still defines the same key."""
    monkeypatch.setenv("KEY", "from-env")
    path = tmp_path / ".env"
    manager = SecretManager([EnvBackend(), DotenvFileBackend(str(path))])

    manager.set("KEY", "from-file", backend_name=f"dotenv:{path}")
    assert manager.get("KEY") == "from-env"  # env still wins
