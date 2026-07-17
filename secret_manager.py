"""secret_manager -- a small, portable, layered secrets resolver.

Standalone and dependency-free: copy this single file into any project. It
has no hard third-party dependencies -- ``keyring``, ``PyYAML``, and
``requests`` are only imported lazily, inside the one backend that needs
each, so a project that doesn't use (say) HashiCorp Vault never needs
``requests`` installed just for this module to import cleanly.

Design
------
A :class:`SecretManager` holds an ordered list of :class:`SecretBackend`
instances (environment variables, OS keychain, a config file, a vault
server, ...) and resolves a key by trying each backend in order, returning
the first non-``None`` result -- the same "explicit override > env var >
file > default" precedence pattern used by most real config systems.

Quickstart
----------
    from secret_manager import SecretManager, EnvBackend, DotenvFileBackend

    secrets = SecretManager([
        EnvBackend(),
        DotenvFileBackend(".env"),
    ])
    api_key = secrets.get("EIA_API_KEY", required=True)

    # Or use the convenience chain (env > OS keychain > dotenv file):
    secrets = SecretManager.default_chain(service_name="my_app", file_path=".env")

Guarding a function on required secrets
----------------------------------------
    @secrets.requires("EIA_API_KEY", "BEA_API_KEY")
    def fetch_energy_and_gdp():
        ...  # only runs if both secrets resolve; raises a clear error otherwise

Writing, updating, and deleting -- any custom key you want
------------------------------------------------------------
There's no fixed schema: every backend accepts an arbitrary key name, and
the writable ones (env, keyring, dotenv, yaml) support create/update/delete
the same way, in place, preserving everything else already stored::

    secrets.set("ANY_CUSTOM_FIELD", "some-value")     # create
    secrets.set("ANY_CUSTOM_FIELD", "new-value")       # update (same call)
    secrets.delete("ANY_CUSTOM_FIELD")                 # remove
    secrets["ANY_CUSTOM_FIELD"] = "some-value"         # dict-style, same as set()

    # Target a specific backend (e.g. this project's own config/config.yaml,
    # which nests settings under a `default:` section):
    secrets.set("NEW_KEY", "value", backend_name="yaml:config/config.yaml",
                section="default")

Extending with your own backend
--------------------------------
Subclass :class:`SecretBackend` and implement :meth:`_fetch`; call
``super().__init__(name=...)`` so the base class's logging/repr machinery
knows what to call your backend::

    class MyBackend(SecretBackend):
        def __init__(self, client):
            super().__init__(name="my-backend")
            self._client = client

        def _fetch(self, key: str) -> str | None:
            return self._client.lookup(key)

No secret value is ever included in a log line, exception message, or
``repr()`` -- see :func:`_mask` / :meth:`SecretManager._redact`.
"""

from __future__ import annotations

import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, TypeVar

log = logging.getLogger("secret_manager")

F = TypeVar("F", bound=Callable[..., object])


class SecretError(Exception):
    """Base class for every error this module raises."""


class SecretNotFoundError(SecretError):
    """Raised when a *required* secret could not be resolved by any backend."""


def _mask(value: str) -> str:
    """Partially mask a secret for safe logging (``'ab***yz'``), never the
    full value. Short values are fully masked rather than partially exposed.
    """
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


class SecretBackend(ABC):
    """One place a secret might live. Template-method base class: subclasses
    only implement :meth:`_fetch`; :meth:`get` supplies the shared logging.

    Parameters
    ----------
    name:
        A short, human-readable label for this backend (used in log lines
        and error messages -- never the secret values themselves).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def _fetch(self, key: str) -> Optional[str]:
        """Look up ``key`` in this backend. Return ``None`` if absent.

        Subclasses implement this; never raise for a simple "not found" --
        reserve exceptions for genuine backend failures (e.g. a vault server
        that's unreachable), so :class:`SecretManager` can still fall
        through to the next backend for the "not configured here" case.
        """

    def get(self, key: str) -> Optional[str]:
        """Resolve ``key`` from this backend, logging the attempt at DEBUG
        level without ever logging the resolved value."""
        value = self._fetch(key)
        self._log_lookup(key, found=value is not None)
        return value

    def _log_lookup(self, key: str, *, found: bool) -> None:
        log.debug("[%s] lookup %r -> %s", self.name, key, "found" if found else "miss")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class EnvBackend(SecretBackend):
    """Reads from process environment variables (12-factor style).

    Parameters
    ----------
    prefix:
        Optional prefix prepended to every lookup, e.g. ``prefix="MYAPP_"``
        so ``get("API_KEY")`` reads ``MYAPP_API_KEY``.
    """

    def __init__(self, prefix: str = "") -> None:
        super().__init__(name="env")
        self._prefix = prefix

    def _fetch(self, key: str) -> Optional[str]:
        return os.environ.get(f"{self._prefix}{key}") or None

    def set(self, key: str, value: str) -> None:
        """Set for the current process (and its children) only -- there is
        no OS-level concept of persisting an env var across process
        restarts, so this does **not** survive a new shell/process."""
        os.environ[f"{self._prefix}{key}"] = value

    def delete(self, key: str) -> None:
        """Unset for the current process only (see :meth:`set`)."""
        os.environ.pop(f"{self._prefix}{key}", None)


class KeyringBackend(SecretBackend):
    """Reads from the OS-native credential store (macOS Keychain, Windows
    Credential Manager, Linux Secret Service) via the optional ``keyring``
    package. Secrets never touch disk in plaintext with this backend.

    Parameters
    ----------
    service_name:
        The "service"/namespace under which secrets were stored, e.g. your
        application's name. Passed straight through to
        ``keyring.get_password(service_name, key)``.
    """

    def __init__(self, service_name: str) -> None:
        super().__init__(name="keyring")
        self._service_name = service_name

    def _fetch(self, key: str) -> Optional[str]:
        try:
            import keyring
        except ImportError:
            log.debug("[%s] 'keyring' package not installed; skipping", self.name)
            return None
        return keyring.get_password(self._service_name, key)

    def set(self, key: str, value: str) -> None:
        """Write (create or overwrite) a secret in the OS keychain."""
        import keyring

        keyring.set_password(self._service_name, key, value)

    def delete(self, key: str) -> None:
        """Remove a secret from the OS keychain.

        Raises whatever ``keyring`` itself raises if ``key`` isn't present
        (typically ``keyring.errors.PasswordDeleteError``) -- this is *not*
        silently idempotent, unlike the file-backend deletes below.
        """
        import keyring

        keyring.delete_password(self._service_name, key)


class FileBackend(SecretBackend):
    """Base class for file-backed secret stores. Subclasses implement
    :meth:`_load` to parse their file format into a ``dict``; this class
    handles caching that parse (a file backend may be asked for many keys)
    and the common "file doesn't exist" case.
    """

    def __init__(self, path: str, *, name: str) -> None:
        super().__init__(name=name)
        self.path = path
        self._cache: Optional[dict[str, str]] = None

    @abstractmethod
    def _load(self) -> dict[str, str]:
        """Parse :attr:`path` into a flat ``{key: value}`` dict."""

    def _fetch(self, key: str) -> Optional[str]:
        if not os.path.isfile(self.path):
            return None
        if self._cache is None:
            self._cache = self._load()
        return self._cache.get(key)

    def reload(self) -> None:
        """Drop the cached parse so the next lookup re-reads the file."""
        self._cache = None


class DotenvFileBackend(FileBackend):
    """Reads ``KEY=value`` pairs from a ``.env``-style file. Stdlib only --
    no dependency on ``python-dotenv``. Blank lines and ``#`` comments are
    ignored; values may be quoted (``KEY="value with spaces"``).
    """

    def __init__(self, path: str = ".env") -> None:
        super().__init__(path, name=f"dotenv:{path}")

    def _load(self) -> dict[str, str]:
        result: dict[str, str] = {}
        with open(self.path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                result.update(self._parse_line(raw_line))
        return result

    @staticmethod
    def _parse_line(raw_line: str) -> dict[str, str]:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            return {}
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        return {key.strip(): value}

    def set(self, key: str, value: str) -> None:
        """Create ``key`` or update it in place if already present, preserving
        every other line (comments, blanks, unrelated keys) untouched."""
        lines = self._read_lines()
        quoted = f'"{value}"' if " " in value else value
        new_line = f"{key}={quoted}\n"

        for i, line in enumerate(lines):
            if key in self._parse_line(line):
                lines[i] = new_line
                break
        else:
            lines.append(new_line)

        self._write_lines(lines)
        self.reload()

    def delete(self, key: str) -> None:
        """Remove ``key``'s line if present; a no-op (not an error) if it
        isn't -- deleting an already-absent key reaches the same end state
        either way."""
        lines = [ln for ln in self._read_lines() if key not in self._parse_line(ln)]
        self._write_lines(lines)
        self.reload()

    def _read_lines(self) -> list[str]:
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            return fh.readlines()

    def _write_lines(self, lines: list[str]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)


class YamlFileBackend(FileBackend):
    """Reads a flat mapping from a YAML file via the optional ``PyYAML``
    package, e.g.::

        EIA_API_KEY: "abc123"
        BEA_API_KEY: "def456"

    A nested document is flattened one level using ``section.key`` if
    ``flatten`` is left ``True`` (the default), so a file like::

        default:
          EIA_API_KEY: "abc123"

    is queryable as either ``"EIA_API_KEY"`` or ``"default.EIA_API_KEY"``.
    """

    def __init__(self, path: str, *, flatten: bool = True) -> None:
        super().__init__(path, name=f"yaml:{path}")
        self._flatten = flatten
        # The as-written nested structure, kept separate from the flattened
        # lookup cache (self._cache) so set()/delete() can round-trip the
        # file without collapsing its original nesting -- e.g. this
        # project's own config.yaml uses a `default: {...}` section, and a
        # naive "dump the flattened dict" would flatten that away.
        self._raw: dict = {}

    def _load(self) -> dict[str, str]:
        import yaml

        with open(self.path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise SecretError(f"{self.path} must be a mapping at the top level")
        self._raw = data
        return self._flatten_dict(data) if self._flatten else dict(data)

    @classmethod
    def _flatten_dict(cls, data: dict, prefix: str = "") -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in data.items():
            qualified = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(cls._flatten_dict(v, prefix=f"{qualified}."))
                # also expose the leaf key unqualified, last-writer-wins
                out.update({kk.rsplit(".", 1)[-1]: vv for kk, vv in
                            cls._flatten_dict(v, prefix="").items()})
            elif v is not None:
                out[qualified] = str(v)
        return out

    def set(self, key: str, value: str, *, section: Optional[str] = None) -> None:
        """Create ``key`` or update it if already present, then rewrite the
        file -- preserving every other key and the original nesting.

        Parameters
        ----------
        section:
            If given, writes under that top-level nested key (e.g.
            ``section="default"`` matches this project's own
            ``config/config.yaml`` layout: ``default: {KEY: value}``).
            Otherwise writes at the top level.
        """
        self._ensure_raw_loaded()
        target = self._raw.setdefault(section, {}) if section else self._raw
        target[key] = value
        self._dump()
        self.reload()

    def delete(self, key: str, *, section: Optional[str] = None) -> None:
        """Remove ``key`` from the given ``section`` (or the top level if
        ``section`` is omitted). A no-op, not an error, if absent there --
        note this only checks the *exact* location given; it does not
        search every nested section for a same-named key."""
        self._ensure_raw_loaded()
        target = self._raw.get(section, {}) if section else self._raw
        target.pop(key, None)
        self._dump()
        self.reload()

    def _ensure_raw_loaded(self) -> None:
        if self._cache is None and os.path.isfile(self.path):
            self._fetch("")  # any key; triggers _load(), which populates _raw

    def _dump(self) -> None:
        import yaml

        with open(self.path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._raw, fh, sort_keys=False, default_flow_style=False)


class VaultBackend(SecretBackend):
    """Reads a single value from a HashiCorp Vault KV v2 secret via the
    optional ``requests`` package.

    Parameters
    ----------
    url:
        Vault server address, e.g. ``"https://vault.internal:8200"``.
    token:
        Vault auth token. Prefer sourcing this itself from an ``EnvBackend``
        rather than hardcoding it.
    mount:
        The KV v2 mount point (default ``"secret"``).
    value_field:
        Which field of the secret's data to return (default ``"value"``) --
        Vault secrets are themselves small key/value maps, so a secret
        written as ``{"value": "abc123"}`` under path ``EIA_API_KEY``
        resolves ``get("EIA_API_KEY")`` to ``"abc123"``.
    """

    def __init__(
        self, url: str, token: str, *, mount: str = "secret", value_field: str = "value"
    ) -> None:
        super().__init__(name="vault")
        self._url = url.rstrip("/")
        self._token = token
        self._mount = mount
        self._value_field = value_field

    def _fetch(self, key: str) -> Optional[str]:
        import requests

        endpoint = f"{self._url}/v1/{self._mount}/data/{key}"
        resp = requests.get(
            endpoint, headers={"X-Vault-Token": self._token}, timeout=10
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("data", {})
        value = data.get(self._value_field)
        return str(value) if value is not None else None

    def set(self, key: str, value: str) -> None:
        """Write ``{value_field: value}`` to ``key``'s KV v2 path.

        NOTE: written against the documented Vault KV v2 HTTP API but not
        exercised against a live Vault server in this environment -- verify
        against your actual Vault instance before relying on it.
        """
        import requests

        endpoint = f"{self._url}/v1/{self._mount}/data/{key}"
        resp = requests.post(
            endpoint,
            headers={"X-Vault-Token": self._token},
            json={"data": {self._value_field: value}},
            timeout=10,
        )
        resp.raise_for_status()

    def delete(self, key: str) -> None:
        """Soft-delete (KV v2's recoverable delete, not a hard destroy) the
        current version at ``key``'s path. Same live-server caveat as
        :meth:`set`."""
        import requests

        endpoint = f"{self._url}/v1/{self._mount}/data/{key}"
        resp = requests.delete(
            endpoint, headers={"X-Vault-Token": self._token}, timeout=10
        )
        if resp.status_code != 404:
            resp.raise_for_status()


class SecretManager:
    """Resolves a secret by trying each :class:`SecretBackend` in order and
    returning the first hit -- highest-precedence backend first.

    Parameters
    ----------
    backends:
        Ordered sequence of backends, highest precedence first.
    cache:
        Cache each resolved (or missing) key in-process, so repeated
        ``get()`` calls for the same key don't re-hit a slow backend
        (keyring prompts, a network call to Vault, ...). Defaults to
        ``True``; call :meth:`clear_cache` if a value can change at runtime.
    """

    def __init__(self, backends: Sequence[SecretBackend], *, cache: bool = True) -> None:
        if not backends:
            raise ValueError("SecretManager needs at least one SecretBackend")
        self._backends = list(backends)
        self._cache_enabled = cache
        self._cache: dict[str, Optional[str]] = {}

    @classmethod
    def default_chain(
        cls,
        *,
        service_name: str,
        env_prefix: str = "",
        file_path: Optional[str] = None,
    ) -> "SecretManager":
        """Build a manager with the recommended precedence for a typical
        project: environment variables, then the OS keychain, then an
        optional dotenv file.

        Parameters
        ----------
        service_name:
            Namespace used for the :class:`KeyringBackend` lookup.
        env_prefix:
            Optional prefix for :class:`EnvBackend` (see its docstring).
        file_path:
            If given, a :class:`DotenvFileBackend` is appended as the
            lowest-precedence backend (used only when neither the
            environment nor the keychain has the key).
        """
        backends: list[SecretBackend] = [
            EnvBackend(prefix=env_prefix),
            KeyringBackend(service_name=service_name),
        ]
        if file_path is not None:
            backends.append(DotenvFileBackend(file_path))
        return cls(backends)

    def get(
        self, key: str, *, default: Optional[str] = None, required: bool = False
    ) -> Optional[str]:
        """Resolve ``key`` across all configured backends.

        Parameters
        ----------
        key:
            The secret's name (backend-specific prefixing, if any, is
            applied inside that backend).
        default:
            Returned if no backend resolves the key and ``required`` is
            ``False``.
        required:
            If ``True``, raise :class:`SecretNotFoundError` instead of
            returning ``default`` when nothing resolves.

        Returns
        -------
        The resolved value, ``default``, or ``None``.

        Raises
        ------
        SecretNotFoundError:
            If ``required`` is ``True`` and no backend resolves ``key``.
        """
        if self._cache_enabled and key in self._cache:
            value = self._cache[key]
        else:
            value = self._resolve(key)
            if self._cache_enabled:
                self._cache[key] = value

        if value is None:
            if required:
                raise SecretNotFoundError(
                    f"{key!r} was not found in any of: "
                    f"{[b.name for b in self._backends]}"
                )
            return default
        return value

    def __getitem__(self, key: str) -> str:
        """Dict-style access always requires the key: ``secrets["FOO"]``."""
        return self.get(key, required=True)  # type: ignore[return-value]

    def _resolve(self, key: str) -> Optional[str]:
        for backend in self._backends:
            value = backend.get(key)
            if value is not None:
                return value
        return None

    def set(self, key: str, value: str, *, backend_name: Optional[str] = None, **kwargs: object) -> None:
        """Create or update ``key`` -> ``value`` in a writable backend.

        Any custom key name works; nothing here is restricted to a
        predefined set of fields.

        Parameters
        ----------
        backend_name:
            Which backend to write to (matches :attr:`SecretBackend.name`,
            e.g. ``"dotenv:.env"``). Defaults to the first backend in the
            precedence chain that supports writing.
        **kwargs:
            Forwarded to that backend's ``set()`` (e.g. ``section=`` for
            :class:`YamlFileBackend`).

        Important
        ---------
        Writing to a *lower*-precedence backend than one that already
        defines ``key`` elsewhere in the chain won't change what
        subsequent :meth:`get` calls return -- a higher-precedence backend
        still wins. That's the correct, expected behavior of a layered
        resolver, not a bug: check :attr:`backend_names` if a write doesn't
        seem to "take".

        Raises
        ------
        SecretError:
            If no configured backend supports writing (or the named one
            doesn't).
        """
        backend = self._find_writable_backend(backend_name, action="write to")
        backend.set(key, value, **kwargs)  # type: ignore[attr-defined]
        self._cache.pop(key, None)  # force the next get() to re-resolve

    def delete(self, key: str, *, backend_name: Optional[str] = None, **kwargs: object) -> None:
        """Remove ``key`` from one writable backend (see :meth:`set` for
        the ``backend_name`` resolution rule and why this only affects one
        backend at a time)."""
        backend = self._find_writable_backend(backend_name, action="delete from")
        backend.delete(key, **kwargs)  # type: ignore[attr-defined]
        self._cache.pop(key, None)

    def __setitem__(self, key: str, value: str) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        self.delete(key)

    def _find_writable_backend(
        self, backend_name: Optional[str], *, action: str
    ) -> SecretBackend:
        candidates = self._backends
        if backend_name is not None:
            candidates = [b for b in self._backends if b.name == backend_name]
            if not candidates:
                raise SecretError(
                    f"No backend named {backend_name!r} configured "
                    f"(have: {self.backend_names})"
                )
        for backend in candidates:
            if hasattr(backend, "set") and hasattr(backend, "delete"):
                return backend
        raise SecretError(
            f"No backend available to {action} -- none of "
            f"{[b.name for b in candidates]} support writing "
            f"(env, keyring, dotenv, and yaml backends all do)"
        )

    def requires(self, *keys: str) -> Callable[[F], F]:
        """Decorator factory: guard a function on one or more secrets being
        resolvable, so it fails fast with a clear error rather than partway
        through its work with a confusing downstream ``KeyError``/``None``.

        Example
        -------
            @secrets.requires("EIA_API_KEY")
            def fetch_energy_prices():
                ...
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                missing = [k for k in keys if self.get(k) is None]
                if missing:
                    raise SecretNotFoundError(
                        f"{func.__name__}() requires secret(s) {missing} which "
                        f"are not configured in any of: "
                        f"{[b.name for b in self._backends]}"
                    )
                return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def clear_cache(self) -> None:
        """Drop all cached lookups (values may have changed at runtime)."""
        self._cache.clear()

    @property
    def backend_names(self) -> tuple[str, ...]:
        """Backend names in precedence order (highest first)."""
        return tuple(b.name for b in self._backends)

    @staticmethod
    def _redact(value: Optional[str]) -> str:
        """Safe-for-logs representation of a possibly-``None`` secret."""
        return "<unset>" if value is None else _mask(value)

    def __repr__(self) -> str:
        cached = {k: self._redact(v) for k, v in self._cache.items()}
        return f"SecretManager(backends={self.backend_names}, cached={cached})"


if __name__ == "__main__":
    # Minimal, dependency-free smoke test / usage demo.
    os.environ["DEMO_API_KEY"] = "supersecretvalue123"

    demo = SecretManager([EnvBackend()])
    print("resolved:", demo._redact(demo.get("DEMO_API_KEY")))
    print("missing, with default:", demo.get("DOES_NOT_EXIST", default="fallback"))

    @demo.requires("DEMO_API_KEY")
    def call_api() -> str:
        return "called!"

    print("guarded call:", call_api())

    try:
        demo.get("DOES_NOT_EXIST", required=True)
    except SecretNotFoundError as exc:
        print("required-and-missing raised as expected:", exc)

    print(demo)

    # --- CRUD against a scratch dotenv file (any custom key name works) ---
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        env_path = os.path.join(tmp, ".env")
        file_demo = SecretManager([DotenvFileBackend(env_path)])

        file_demo.set("A_BRAND_NEW_CUSTOM_FIELD", "first-value")
        print("after create:", file_demo.get("A_BRAND_NEW_CUSTOM_FIELD"))

        file_demo.set("A_BRAND_NEW_CUSTOM_FIELD", "updated-value")
        print("after update:", file_demo.get("A_BRAND_NEW_CUSTOM_FIELD"))

        file_demo.delete("A_BRAND_NEW_CUSTOM_FIELD")
        print("after delete:", file_demo.get("A_BRAND_NEW_CUSTOM_FIELD", default="<gone>"))
