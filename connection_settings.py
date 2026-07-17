"""connection_settings -- named data connection profiles backed by secrets.

This module sits one level above ``secret_manager.py``. The secret manager
already knows how to resolve and persist flat key/value pairs across env vars,
keyring, dotenv, YAML, and Vault. ``ConnectionSettingsManager`` gives those
flat keys a small schema: named profiles, common provider templates, required
field validation, and safe redacted views for logs.

Quickstart
----------
    from secret_manager import SecretManager, EnvBackend, DotenvFileBackend
    from connection_settings import ConnectionSettingsManager

    secrets = SecretManager([EnvBackend(), DotenvFileBackend(".connections.env")])
    connections = ConnectionSettingsManager(secrets)

    connections.set_connection(
        "warehouse",
        "databricks",
        {
            "server_hostname": "dbc-123.cloud.databricks.com",
            "http_path": "/sql/1.0/warehouses/abc",
            "access_token": "dapi...",
            "catalog": "main",
            "schema": "analytics",
        },
    )

    profile = connections.get_connection("warehouse")
    print(profile.as_dict(redact=True))

The storage keys are intentionally plain and portable:
``CONNECTION__WAREHOUSE__KIND``, ``CONNECTION__WAREHOUSE__HOST``, etc. Env vars
can override file/keyring/Vault values through the normal SecretManager
precedence chain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote_plus

from secret_manager import SecretError, SecretManager, _mask

UriBuilder = Callable[["ConnectionProfile"], Optional[str]]


class ConnectionSettingsError(SecretError):
    """Base error for connection profile validation/resolution failures."""


@dataclass(frozen=True)
class SettingField:
    """One field in a connection template.

    ``secret=True`` marks values that should be redacted in ``repr`` and
    ``as_dict(redact=True)``. It does not force a storage backend; use the
    SecretManager chain to decide whether secrets live in keyring, Vault,
    dotenv, YAML, or process env vars.
    """

    name: str
    required: bool = True
    secret: bool = False
    default: Optional[str] = None
    description: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectionTemplate:
    """Reusable provider schema for a class of data connection."""

    kind: str
    fields: tuple[SettingField, ...]
    aliases: tuple[str, ...] = ()
    required_any: tuple[tuple[str, ...], ...] = ()
    uri_builder: Optional[UriBuilder] = None
    description: str = ""

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def secret_fields(self) -> frozenset[str]:
        return frozenset(field.name for field in self.fields if field.secret)

    def field_for(self, name: str) -> Optional[SettingField]:
        canonical = _normalize_field_name(name)
        for item in self.fields:
            names = (item.name, *item.aliases)
            if canonical in {_normalize_field_name(candidate) for candidate in names}:
                return item
        return None

    def canonical_field_name(self, name: str) -> str:
        spec = self.field_for(name)
        return spec.name if spec is not None else _normalize_field_name(name)


@dataclass(frozen=True)
class ConnectionProfile:
    """Resolved values for one named connection."""

    name: str
    kind: str
    values: Mapping[str, str]
    template: ConnectionTemplate
    secret_fields: frozenset[str] = field(default_factory=frozenset)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.values.get(self.template.canonical_field_name(key), default)

    def as_dict(self, *, redact: bool = False) -> dict[str, str]:
        if not redact:
            return dict(self.values)
        return {
            key: _redact(value) if key in self.secret_fields else value
            for key, value in self.values.items()
        }

    def connection_uri(self, *, redact: bool = False) -> Optional[str]:
        if self.template.uri_builder is None:
            return None
        uri = self.template.uri_builder(self)
        if redact and uri is not None:
            for key in self.secret_fields:
                value = self.values.get(key)
                if value:
                    uri = uri.replace(quote_plus(value), "***").replace(value, "***")
        return uri

    def __repr__(self) -> str:
        return (
            f"ConnectionProfile(name={self.name!r}, kind={self.kind!r}, "
            f"values={self.as_dict(redact=True)!r})"
        )


class ConnectionSettingsManager:
    """Store and resolve named connection profiles using ``SecretManager``.

    The manager stores each profile as flat keys with a configurable namespace.
    This means it works with every existing SecretManager backend and keeps env
    override behavior simple and inspectable.
    """

    def __init__(
        self,
        secrets: SecretManager,
        *,
        namespace: str = "CONNECTION",
        templates: Optional[Iterable[ConnectionTemplate]] = None,
    ) -> None:
        self.secrets = secrets
        self.namespace = _normalize_storage_part(namespace)
        self._templates = _build_template_registry(templates or COMMON_TEMPLATES)

    @property
    def template_names(self) -> tuple[str, ...]:
        return tuple(sorted({template.kind for template in self._templates.values()}))

    def template_for(self, kind: str) -> ConnectionTemplate:
        key = _normalize_kind(kind)
        try:
            return self._templates[key]
        except KeyError as exc:
            raise ConnectionSettingsError(
                f"Unknown connection kind {kind!r}. Available kinds: {self.template_names}"
            ) from exc

    def storage_key(self, profile_name: str, field_name: str) -> str:
        return "__".join(
            [
                self.namespace,
                _normalize_storage_part(profile_name),
                _normalize_storage_part(field_name),
            ]
        )

    def set_connection(
        self,
        name: str,
        kind: str,
        values: Mapping[str, object],
        *,
        backend_name: Optional[str] = None,
        validate: bool = True,
        **backend_kwargs: object,
    ) -> ConnectionProfile:
        """Create or update a connection profile.

        Unknown fields are allowed and stored, so teams can add provider
        settings without changing the module. Built-in fields get canonical
        names and validation.
        """
        template = self.template_for(kind)
        canonical_values = {
            template.canonical_field_name(key): str(value)
            for key, value in values.items()
            if value is not None
        }
        if validate:
            self._validate(name, template, canonical_values)

        field_names = self._field_names_for_write(name, template, canonical_values)
        self.secrets.set(
            self.storage_key(name, "KIND"),
            template.kind,
            backend_name=backend_name,
            **backend_kwargs,
        )
        self.secrets.set(
            self.storage_key(name, "FIELDS"),
            ",".join(field_names),
            backend_name=backend_name,
            **backend_kwargs,
        )
        for key, value in canonical_values.items():
            self.secrets.set(
                self.storage_key(name, key),
                value,
                backend_name=backend_name,
                **backend_kwargs,
            )
        self.secrets.clear_cache()
        return self.get_connection(name)

    def get_connection(
        self, name: str, *, required: bool = True
    ) -> Optional[ConnectionProfile]:
        kind = self.secrets.get(self.storage_key(name, "KIND"), required=required)
        if kind is None:
            if required:
                raise ConnectionSettingsError(
                    f"Connection profile {name!r} is not configured"
                )
            return None

        template = self.template_for(kind)
        field_names = self._field_names_for_read(name, template)
        values: dict[str, str] = {}
        for field_name in field_names:
            value = self.secrets.get(self.storage_key(name, field_name))
            spec = template.field_for(field_name)
            if value is None and spec is not None:
                value = spec.default
            if value is not None:
                values[field_name] = value

        self._validate(name, template, values)
        return ConnectionProfile(
            name=name,
            kind=template.kind,
            values=values,
            template=template,
            secret_fields=self._secret_fields(template, field_names),
        )

    def delete_connection(
        self,
        name: str,
        *,
        backend_name: Optional[str] = None,
        **backend_kwargs: object,
    ) -> None:
        kind = self.secrets.get(self.storage_key(name, "KIND"))
        template = self.template_for(kind) if kind is not None else None
        field_names = self._field_names_for_read(name, template) if template else ()
        for field_name in ("KIND", "FIELDS", *field_names):
            self.secrets.delete(
                self.storage_key(name, field_name),
                backend_name=backend_name,
                **backend_kwargs,
            )
        self.secrets.clear_cache()

    def connection_uri(self, name: str, *, redact: bool = False) -> Optional[str]:
        profile = self.get_connection(name)
        return profile.connection_uri(redact=redact) if profile is not None else None

    def _field_names_for_write(
        self,
        name: str,
        template: ConnectionTemplate,
        values: Mapping[str, str],
    ) -> tuple[str, ...]:
        existing = _split_csv(self.secrets.get(self.storage_key(name, "FIELDS")))
        return _dedupe((*template.field_names, *existing, *values.keys()))

    def _field_names_for_read(
        self,
        name: str,
        template: ConnectionTemplate,
    ) -> tuple[str, ...]:
        stored = _split_csv(self.secrets.get(self.storage_key(name, "FIELDS")))
        return _dedupe((*template.field_names, *stored))

    def _validate(
        self,
        name: str,
        template: ConnectionTemplate,
        values: Mapping[str, str],
    ) -> None:
        materialized = dict(values)
        for spec in template.fields:
            if spec.name not in materialized:
                current = self.secrets.get(self.storage_key(name, spec.name))
                if current is not None:
                    materialized[spec.name] = current
                elif spec.default is not None:
                    materialized[spec.name] = spec.default

        missing = [
            spec.name
            for spec in template.fields
            if spec.required and not materialized.get(spec.name)
        ]
        if template.required_any and not any(
            all(materialized.get(field_name) for field_name in group)
            for group in template.required_any
        ):
            choices = [" + ".join(group) for group in template.required_any]
            missing.append("one of: " + " OR ".join(choices))
        if missing:
            raise ConnectionSettingsError(
                f"Connection profile {name!r} ({template.kind}) is missing required "
                f"setting(s): {missing}"
            )

    @staticmethod
    def _secret_fields(
        template: ConnectionTemplate, field_names: Sequence[str]
    ) -> frozenset[str]:
        secret_fields = set(template.secret_fields)
        for field_name in field_names:
            lowered = field_name.lower()
            if any(token in lowered for token in ("password", "secret", "token", "key")):
                secret_fields.add(field_name)
        return frozenset(secret_fields)


def _build_template_registry(
    templates: Iterable[ConnectionTemplate],
) -> dict[str, ConnectionTemplate]:
    registry: dict[str, ConnectionTemplate] = {}
    for template in templates:
        names = (template.kind, *template.aliases)
        for name in names:
            registry[_normalize_kind(name)] = template
    return registry


def _normalize_kind(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalize_storage_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
    if not normalized:
        raise ConnectionSettingsError("Storage key parts cannot be blank")
    return normalized


def _split_csv(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _redact(value: str) -> str:
    return _mask(value)


def _quoted(value: Optional[str]) -> str:
    return quote_plus(value or "")


def _host_port(profile: ConnectionProfile, default_port: str) -> str:
    host = profile.get("host", "")
    port = profile.get("port", default_port)
    return f"{host}:{port}" if port else host or ""


def _sql_uri(scheme: str, default_port: str) -> UriBuilder:
    def build(profile: ConnectionProfile) -> Optional[str]:
        username = _quoted(profile.get("username"))
        password = _quoted(profile.get("password"))
        auth = f"{username}:{password}@" if username or password else ""
        database = _quoted(profile.get("database"))
        return f"{scheme}://{auth}{_host_port(profile, default_port)}/{database}"

    return build


def _sqlite_uri(profile: ConnectionProfile) -> Optional[str]:
    path = profile.get("path")
    return f"sqlite:///{path}" if path else None


def _oracle_uri(profile: ConnectionProfile) -> Optional[str]:
    dsn = profile.get("dsn")
    username = _quoted(profile.get("username"))
    password = _quoted(profile.get("password"))
    if dsn:
        return f"oracle+oracledb://{username}:{password}@{dsn}"
    service = _quoted(profile.get("service_name"))
    return f"oracle+oracledb://{username}:{password}@{_host_port(profile, '1521')}/{service}"


def _mssql_uri(profile: ConnectionProfile) -> Optional[str]:
    base = _sql_uri("mssql+pyodbc", "1433")(profile)
    driver = quote_plus(profile.get("driver", "ODBC Driver 18 for SQL Server") or "")
    trust = profile.get("trust_server_certificate")
    params = [f"driver={driver}"] if driver else []
    if trust:
        params.append(f"TrustServerCertificate={quote_plus(trust)}")
    return f"{base}?{'&'.join(params)}" if params else base


def _databricks_uri(profile: ConnectionProfile) -> Optional[str]:
    host = profile.get("server_hostname")
    http_path = quote_plus(profile.get("http_path", "") or "")
    token = _quoted(profile.get("access_token"))
    if not host:
        return None
    params = []
    for key in ("catalog", "schema", "warehouse_id"):
        value = profile.get(key)
        if value:
            params.append(f"{key}={quote_plus(value)}")
    query = f"?{'&'.join(params)}" if params else ""
    return f"databricks://token:{token}@{host}:443/{http_path}{query}"


def _snowflake_uri(profile: ConnectionProfile) -> Optional[str]:
    username = _quoted(profile.get("username"))
    password = _quoted(profile.get("password"))
    account = profile.get("account", "")
    database = _quoted(profile.get("database"))
    schema = _quoted(profile.get("schema"))
    params = []
    for key in ("warehouse", "role"):
        value = profile.get(key)
        if value:
            params.append(f"{key}={quote_plus(value)}")
    query = f"?{'&'.join(params)}" if params else ""
    return f"snowflake://{username}:{password}@{account}/{database}/{schema}{query}"


def _field(
    name: str,
    *,
    required: bool = True,
    secret: bool = False,
    default: Optional[str] = None,
    aliases: Sequence[str] = (),
    description: str = "",
) -> SettingField:
    return SettingField(
        name=name,
        required=required,
        secret=secret,
        default=default,
        aliases=tuple(aliases),
        description=description,
    )


COMMON_TEMPLATES: tuple[ConnectionTemplate, ...] = (
    ConnectionTemplate(
        "postgresql",
        aliases=("postgres", "pg"),
        uri_builder=_sql_uri("postgresql", "5432"),
        fields=(
            _field("host"),
            _field("port", required=False, default="5432"),
            _field("database", aliases=("db",)),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
            _field("sslmode", required=False, default="prefer"),
        ),
    ),
    ConnectionTemplate(
        "mysql",
        aliases=("mariadb",),
        uri_builder=_sql_uri("mysql", "3306"),
        fields=(
            _field("host"),
            _field("port", required=False, default="3306"),
            _field("database", aliases=("db",)),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
        ),
    ),
    ConnectionTemplate(
        "mssql",
        aliases=("sql", "sqlserver", "sql_server", "sql-server"),
        uri_builder=_mssql_uri,
        fields=(
            _field("host"),
            _field("port", required=False, default="1433"),
            _field("database", aliases=("db",)),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
            _field("driver", required=False, default="ODBC Driver 18 for SQL Server"),
            _field("trust_server_certificate", required=False, default="no"),
        ),
    ),
    ConnectionTemplate(
        "oracle",
        uri_builder=_oracle_uri,
        required_any=(("dsn",), ("host", "service_name")),
        fields=(
            _field("dsn", required=False),
            _field("host", required=False),
            _field("port", required=False, default="1521"),
            _field("service_name", required=False, aliases=("service", "sid")),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
        ),
    ),
    ConnectionTemplate(
        "databricks",
        aliases=("dbx", "databricks_sql"),
        uri_builder=_databricks_uri,
        fields=(
            _field("server_hostname", aliases=("host", "hostname")),
            _field("http_path"),
            _field("access_token", secret=True, aliases=("token", "pat")),
            _field("catalog", required=False),
            _field("schema", required=False),
            _field("warehouse_id", required=False),
        ),
    ),
    ConnectionTemplate(
        "sybase",
        uri_builder=_sql_uri("sybase", "5000"),
        fields=(
            _field("host"),
            _field("port", required=False, default="5000"),
            _field("database", aliases=("db",)),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
        ),
    ),
    ConnectionTemplate(
        "snowflake",
        uri_builder=_snowflake_uri,
        fields=(
            _field("account"),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
            _field("warehouse", required=False),
            _field("database", aliases=("db",)),
            _field("schema", required=False),
            _field("role", required=False),
        ),
    ),
    ConnectionTemplate(
        "redshift",
        uri_builder=_sql_uri("redshift+psycopg2", "5439"),
        fields=(
            _field("host"),
            _field("port", required=False, default="5439"),
            _field("database", aliases=("db",)),
            _field("username", aliases=("user",)),
            _field("password", secret=True),
            _field("sslmode", required=False, default="prefer"),
        ),
    ),
    ConnectionTemplate(
        "sqlite",
        uri_builder=_sqlite_uri,
        fields=(_field("path"),),
    ),
    ConnectionTemplate(
        "aws",
        aliases=("amazon_web_services",),
        fields=(
            _field("access_key_id", required=False, secret=True),
            _field("secret_access_key", required=False, secret=True),
            _field("session_token", required=False, secret=True),
            _field("region", required=False, default="us-east-1"),
            _field("profile", required=False),
            _field("role_arn", required=False),
        ),
    ),
    ConnectionTemplate(
        "s3",
        aliases=("aws_s3",),
        fields=(
            _field("bucket"),
            _field("region", required=False, default="us-east-1"),
            _field("prefix", required=False),
            _field("endpoint_url", required=False),
            _field("access_key_id", required=False, secret=True),
            _field("secret_access_key", required=False, secret=True),
            _field("session_token", required=False, secret=True),
        ),
    ),
    ConnectionTemplate(
        "odbc",
        fields=(
            _field("dsn"),
            _field("username", required=False, aliases=("user",)),
            _field("password", required=False, secret=True),
            _field("driver", required=False),
        ),
    ),
    ConnectionTemplate(
        "jdbc",
        fields=(
            _field("jdbc_url"),
            _field("username", required=False, aliases=("user",)),
            _field("password", required=False, secret=True),
            _field("driver", required=False),
        ),
    ),
)
