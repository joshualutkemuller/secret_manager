"""Tests for connection_settings.py.

Run standalone: `python -m pytest tools/test_connection_settings.py -v`
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from connection_settings import (  # noqa: E402
    ConnectionSettingsError,
    ConnectionSettingsManager,
)
from secret_manager import DotenvFileBackend, EnvBackend, SecretManager  # noqa: E402


def test_common_templates_include_requested_connection_types(tmp_path):
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(tmp_path / ".env"))]))
    names = set(manager.template_names)
    assert {"oracle", "mssql", "databricks", "sybase", "aws"}.issubset(names)


def test_set_and_get_mssql_profile_with_defaults_and_redaction(tmp_path):
    path = tmp_path / ".env"
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(path))]))

    profile = manager.set_connection(
        "prod sql",
        "sql",
        {
            "host": "sql.internal",
            "database": "warehouse",
            "user": "svc",
            "password": "supersecretvalue123",
        },
    )

    assert profile.kind == "mssql"
    assert profile.get("username") == "svc"
    assert profile.get("port") == "1433"
    assert profile.as_dict(redact=True)["password"] == "su***************23"
    assert "supersecretvalue123" not in repr(profile)
    assert "supersecretvalue123" not in profile.connection_uri(redact=True)
    assert manager.storage_key("prod sql", "password") in path.read_text()


def test_environment_override_wins_over_file_backend(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    secrets = SecretManager([EnvBackend(), DotenvFileBackend(str(path))])
    manager = ConnectionSettingsManager(secrets)
    manager.set_connection(
        "warehouse",
        "postgresql",
        {
            "host": "from-file",
            "database": "analytics",
            "username": "svc",
            "password": "from-file-password",
        },
        backend_name=f"dotenv:{path}",
    )

    monkeypatch.setenv(manager.storage_key("warehouse", "host"), "from-env")
    secrets.clear_cache()

    assert manager.get_connection("warehouse").get("host") == "from-env"


def test_databricks_profile_materializes_expected_fields(tmp_path):
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(tmp_path / ".env"))]))
    profile = manager.set_connection(
        "lakehouse",
        "databricks",
        {
            "host": "dbc.example.cloud.databricks.com",
            "http_path": "/sql/1.0/warehouses/abc",
            "token": "dapi-secret",
            "catalog": "main",
            "schema": "gold",
        },
    )

    assert profile.get("server_hostname") == "dbc.example.cloud.databricks.com"
    assert profile.get("access_token") == "dapi-secret"
    assert profile.as_dict(redact=True)["access_token"] == "da*******et"
    assert profile.connection_uri(redact=True).startswith("databricks://token:***@")


def test_oracle_accepts_dsn_instead_of_host_and_service(tmp_path):
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(tmp_path / ".env"))]))
    profile = manager.set_connection(
        "finance-oracle",
        "oracle",
        {
            "dsn": "finance_high",
            "username": "svc",
            "password": "oracle-password",
        },
    )

    assert profile.get("dsn") == "finance_high"
    assert "finance_high" in profile.connection_uri()


def test_missing_required_setting_raises(tmp_path):
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(tmp_path / ".env"))]))

    with pytest.raises(ConnectionSettingsError, match="password"):
        manager.set_connection(
            "bad",
            "sybase",
            {"host": "sybase.internal", "database": "risk", "username": "svc"},
        )


def test_get_connection_can_return_none_when_optional(tmp_path):
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(tmp_path / ".env"))]))

    assert manager.get_connection("missing", required=False) is None


def test_custom_fields_are_stored_and_returned(tmp_path):
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(tmp_path / ".env"))]))
    profile = manager.set_connection(
        "aws-prod",
        "aws",
        {"profile": "prod", "region": "us-west-2", "external_id": "abc123"},
    )

    assert profile.get("profile") == "prod"
    assert profile.get("external_id") == "abc123"


def test_delete_connection_removes_profile_keys(tmp_path):
    path = tmp_path / ".env"
    manager = ConnectionSettingsManager(SecretManager([DotenvFileBackend(str(path))]))
    manager.set_connection("local", "sqlite", {"path": "local.db"})

    manager.delete_connection("local")

    with pytest.raises(Exception):
        manager.get_connection("local")
    assert "LOCAL" not in path.read_text()
