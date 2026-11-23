# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 ClaymoreLab
"""Local OAuth registration and token ownership for Xia Director."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import threading
import time
import uuid
import webbrowser
from base64 import urlsafe_b64encode
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from ipaddress import IPv6Address, ip_address
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
from portalocker import Lock
from portalocker.exceptions import LockException

APP_NAME = "DramaClaw"
PROVIDER = "openai_chatgpt_plan"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SHARING_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {SHARING_SCOPE}"
DYNAMIC_CLIENT = "dynamic_agent_client"
CALLBACK_PATH = "/auth/callback"
USAGE_URL = "https://chatgpt.com/settings/usage"
# Ultra can delegate work automatically. Director keeps Codex delegation disabled.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_REFRESH_FAILURES = {
    "invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused",
}


class PlanError(RuntimeError):
    """Safe to return to the editor. Never contains upstream bodies or tokens."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _sign_in_required() -> PlanError:
    return PlanError("reauth_required", "Sign in with ChatGPT again in Settings → Models & Channels → Agent model.")


def _permission_required() -> PlanError:
    return PlanError("permission_required", "Enable ChatGPT plan usage for DramaClaw and grant consent in your browser.")


def is_local_connection(connection: Any) -> bool:
    host = connection.url.hostname
    origin = connection.headers.get("origin")
    if host not in {"localhost", "127.0.0.1", "::1"} or connection.client is None:
        return False
    try:
        peer = ip_address(connection.client.host)
    except ValueError:
        return False
    if isinstance(peer, IPv6Address) and peer.ipv4_mapped is not None:
        peer = peer.ipv4_mapped
    if not peer.is_loopback:
        return False
    if origin is None:
        return True
    parts = urlsplit(origin)
    return parts.scheme in {"http", "https"} and parts.netloc == connection.url.netloc


@dataclass
class Authorization:
    redirect_uri: str
    client_id: str | None = None
    state: str = field(default_factory=secrets.token_urlsafe, repr=False)
    nonce: str = field(default_factory=secrets.token_urlsafe, repr=False)
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(48), repr=False)

    def url(self, host_id: str, profile: dict | None, *, reconsent: bool = False) -> str:
        challenge = urlsafe_b64encode(hashlib.sha256(self.verifier.encode()).digest())
        query = {
            "client_id": self.client_id or DYNAMIC_CLIENT,
            "ext_agent_host_id": host_id,
            "redirect_uri": self.redirect_uri,
            "resource": RESOURCE,
            "response_type": "code",
            "scope": SCOPES,
            "state": self.state,
            "nonce": self.nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
        }
        if self.client_id:
            if profile and profile.get("id_token"):
                query["id_token_hint"] = profile["id_token"]
            if profile and profile.get("email"):
                query["login_hint"] = profile["email"]
            if reconsent:
                query["prompt"] = "consent"
        else:
            query["agent_name_hint"] = APP_NAME
        return f"{ISSUER}/api/accounts/authorize?{urlencode(query)}"

    def callback(self, query: dict[str, list[str]]) -> tuple[str, str]:
        def one(name: str) -> str:
            values = query.get(name, [])
            return values[0] if len(values) == 1 else ""

        if not secrets.compare_digest(one("state").encode(), self.state.encode()):
            raise PlanError("invalid_callback", "ChatGPT sign-in returned an invalid state.")
        if any(len(query.get(name, [])) > 1 for name in ("client_id", "error")):
            raise PlanError("invalid_callback", "OpenAI returned an ambiguous ChatGPT sign-in result.")
        if one("error"):
            if one("error") == "access_denied":
                raise _permission_required()
            raise PlanError("sign_in_failed", "OpenAI could not complete ChatGPT sign-in. Check account and workspace access.")
        issued = one("client_id") or self.client_id
        if not issued or not issued.startswith("oaiapp_") or (
            self.client_id and issued != self.client_id
        ) or not one("code"):
            raise PlanError("invalid_callback", "OpenAI did not return a valid ChatGPT registration. Try signing in again.")
        return one("code"), issued


class _CallbackHandler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        self.request.settimeout(3)
        super().setup()

    def log_message(self, *_args: Any) -> None:
        # The callback URL contains the OAuth authorization code.
        pass

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path != CALLBACK_PATH or self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(404)
            return
        query = parse_qs(parsed.query, max_num_fields=16)
        attempt = self.server.attempt
        if query.get("state") != [attempt.state]:
            self.send_error(400, "Invalid sign-in state")
            return
        self.server.callback = query
        message = b"Sign-in response received. Return to DramaClaw to check your connection."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(message)))
        self.end_headers()
        self.wfile.write(message)


class _CallbackServer(HTTPServer):
    timeout = 0.5
    callback: dict[str, list[str]] | None = None

    def handle_error(self, *_args: Any) -> None:
        # A malformed browser request must not print a callback or traceback.
        pass


def _auth_endpoint(url: Any) -> str:
    if isinstance(url, str):
        parts = urlsplit(url)
        if parts.scheme == "https" and parts.netloc == "auth.openai.com" and not parts.fragment:
            return url
    raise PlanError("invalid_discovery", "OpenAI returned invalid ChatGPT sign-in metadata.")


def _refresh_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return 0


def _clear_tokens(profile: dict, status: str) -> None:
    for key in (
        "id_token", "access_token", "refresh_token", "expires_at",
        "refresh_at", "refresh_needed", "understanding_connection_id",
    ):
        profile.pop(key, None)
    profile["status"] = status


class PlanSession:
    def __init__(self, directory: Path, *, transport: httpx.BaseTransport | None = None):
        self.directory = directory
        self.path = directory / "session.json"
        self._transport = transport
        self._browser_lock = threading.Lock()
        self._pending = False
        self._notice: PlanError | None = None

    @contextmanager
    def locked_state(self):
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = self.directory.lstat()
            if self.directory.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise OSError("insecure directory")
            with Lock(str(self.directory / "session.lock"), mode="a", timeout=25):
                if self.path.exists() or self.path.is_symlink():
                    info = self.path.lstat()
                    if self.path.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
                        raise OSError("insecure credential file")
                    state = json.loads(self.path.read_text(encoding="utf-8"))
                    if not (
                        isinstance(state, dict)
                        and isinstance(state.get("profiles"), dict)
                        and isinstance(state.get("host_id"), str)
                        and "active" in state
                    ):
                        raise ValueError("invalid credential state")
                else:
                    state = {"active": None, "profiles": {}, "host_id": f"urn:uuid:{uuid.uuid4()}"}
                yield state
        except (OSError, ValueError, TypeError, LockException):
            raise PlanError("storage_error", "DramaClaw could not access its protected ChatGPT credential storage.") from None

    def save(self, state: dict) -> None:
        temp_path = None
        try:
            with NamedTemporaryFile(mode="w", dir=self.directory, encoding="utf-8", delete=False) as handle:
                temp_path = Path(handle.name)
                json.dump(state, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(self.path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            with httpx.Client(timeout=12, trust_env=False, transport=self._transport) as client:
                return client.request(method, url, **kwargs)
        except httpx.HTTPError:
            raise PlanError("connection_failed", "Could not reach OpenAI for ChatGPT plan access. Check your connection and try again.") from None

    def _metadata(self) -> dict:
        response = self._request("GET", f"{ISSUER}/.well-known/openid-configuration")
        try:
            data = response.json()
            if response.is_success and data["issuer"] == ISSUER:
                return data
        except (ValueError, KeyError, TypeError):
            pass
        raise PlanError("invalid_discovery", "OpenAI sign-in metadata is unavailable. Try again later.")

    def _exchange(self, **form: str) -> dict:
        response = self._request("POST", f"{ISSUER}/api/accounts/oauth/token", data={**form, "resource": RESOURCE})
        try:
            result = response.json()
        except ValueError:
            result = None
        if response.is_success and isinstance(result, dict):
            return result
        code = result.get("error") if isinstance(result, dict) else None
        if code in ("invalid_client", *_REFRESH_FAILURES):
            raise PlanError(code, "OpenAI rejected the ChatGPT sign-in session.")
        raise PlanError("token_exchange_failed", "OpenAI could not renew or complete ChatGPT sign-in. Try again later.")

    def _identity(self, token: str, client_id: str, nonce: str) -> dict:
        try:
            endpoint = _auth_endpoint(self._metadata().get("jwks_uri"))
            response = self._request("GET", endpoint)
            if not response.is_success:
                raise PlanError("identity_unavailable", "OpenAI sign-in verification is temporarily unavailable.")
            header = jwt.get_unverified_header(token)
            keys = jwt.PyJWKSet.from_dict(response.json()).keys
            key = next(key for key in keys if key.key_id == header.get("kid"))
            claims = jwt.decode(
                token, key.key, algorithms=["RS256"], issuer=ISSUER, audience=client_id,
                options={"require": ["exp", "iss", "aud", "sub", "nonce"]},
            )
            audience = claims["aud"]
            authorized_party = claims.get("azp")
            if (
                claims["nonce"] != nonce or not claims["sub"]
                or (authorized_party is not None and authorized_party != client_id)
                or (isinstance(audience, list) and len(audience) > 1 and authorized_party != client_id)
            ):
                raise ValueError
            return claims
        except (jwt.PyJWTError, ValueError, TypeError, StopIteration):
            raise PlanError("invalid_identity", "DramaClaw could not verify the ChatGPT account. Sign in again.") from None

    @staticmethod
    def _tokens(data: dict, *, previous_scopes: list[str] | None = None) -> dict:
        try:
            scopes = data["scope"].split() if "scope" in data else previous_scopes
            expires = float(data["expires_in"])
            access = data["access_token"]
            refresh = data.get("refresh_token")
            if (
                not scopes or not math.isfinite(expires) or expires <= 0
                or not isinstance(access, str) or not access
                or ("offline_access" in scopes and (not isinstance(refresh, str) or not refresh))
                or data["token_type"].lower() != "bearer"
            ):
                raise ValueError
            return {
                "access_token": access, "refresh_token": refresh,
                "expires_at": time.time() + expires,
                "refresh_at": _refresh_time(data.get("earliest_refresh_at")),
                "scopes": scopes, "status": "connected", "refresh_needed": False,
            }
        except (KeyError, AttributeError, TypeError, ValueError):
            raise PlanError("invalid_token_response", "OpenAI returned incomplete ChatGPT credentials. Sign in again.") from None

    @staticmethod
    def _active_profile(state: dict) -> dict:
        return state["profiles"].get(state["active"], {})

    def status(self) -> dict:
        with self.locked_state() as state:
            active = self._active_profile(state)
            status = active.get("status", "disconnected")
            sharing = SHARING_SCOPE in active.get("scopes", []) and status == "connected"
            with self._browser_lock:
                notice = self._notice
                pending = self._pending
            return {
                "status": "connecting" if pending else status,
                "sharing": sharing,
                "activeProfileId": state["active"],
                "profiles": [
                    {"id": key, "label": value["label"], "email": value.get("email", "")}
                    for key, value in state["profiles"].items()
                    if "label" in value
                ],
                "message": str(notice) if notice else None,
            }

    def start_sign_in(self, profile_id: str | None, *, reconsent: bool = False) -> None:
        with self._browser_lock:
            if self._pending:
                raise PlanError("connection_busy", "Complete the open ChatGPT sign-in in your browser.")
            self._pending = True
            self._notice = None
        threading.Thread(
            target=self._authorize, args=(profile_id, reconsent), daemon=True,
            name="director-chatgpt-oauth",
        ).start()

    def _authorize(self, profile_id: str | None, reconsent: bool) -> None:
        notice = None
        try:
            with self.locked_state() as state:
                if profile_id and profile_id not in state["profiles"]:
                    raise PlanError("unknown_profile", "Choose an existing ChatGPT connection.")
                pending_id = state.get("pending_registration")
                key = profile_id or pending_id or uuid.uuid4().hex
                previous = state["profiles"].get(key)
                host_id = state["host_id"]
                self.save(state)
            client_id = previous.get("client_id") if previous else None
            for attempt_number in range(2):
                with _CallbackServer(("127.0.0.1", 0), _CallbackHandler) as server:
                    redirect = f"http://127.0.0.1:{server.server_port}{CALLBACK_PATH}"
                    attempt = server.attempt = Authorization(redirect, client_id)
                    if not webbrowser.open(attempt.url(host_id, previous, reconsent=reconsent)):
                        raise PlanError("browser_unavailable", "DramaClaw could not open your default browser.")
                    deadline = time.monotonic() + 300
                    while server.callback is None and time.monotonic() < deadline:
                        server.handle_request()
                    if server.callback is None:
                        raise PlanError("sign_in_timed_out", "ChatGPT browser sign-in timed out. Try again.")
                    code, client_id = attempt.callback(server.callback)
                if previous is None:
                    with self.locked_state() as state:
                        previous = {"client_id": client_id}
                        state["profiles"][key] = previous
                        state["pending_registration"] = key
                        self.save(state)
                try:
                    data = self._exchange(
                        client_id=client_id, grant_type="authorization_code", code=code,
                        code_verifier=attempt.verifier, redirect_uri=redirect,
                    )
                    break
                except PlanError as exc:
                    if exc.code != "invalid_grant" or attempt_number:
                        raise
            id_token = data.get("id_token")
            if not isinstance(id_token, str) or not id_token:
                raise PlanError("invalid_identity", "OpenAI did not return a ChatGPT account identity.")
            claims = self._identity(id_token, client_id, attempt.nonce)
            if previous.get("subject") and previous["subject"] != claims["sub"]:
                raise PlanError("account_mismatch", "Choose the ChatGPT account for this saved connection.")
            tokens = self._tokens(data)
            with self.locked_state() as state:
                label = previous.get("label")
                if label is None:
                    label = f"Connection {1 + sum('label' in item for item in state['profiles'].values())}"
                state["profiles"][key] = {
                    **tokens, "label": label, "client_id": client_id, "id_token": id_token,
                    "email": claims.get("email", ""), "subject": claims["sub"],
                }
                state["active"] = key
                if state.get("pending_registration") == key:
                    state.pop("pending_registration")
                self.save(state)
        except PlanError as exc:
            notice = exc
        except Exception:
            notice = PlanError("sign_in_failed", "DramaClaw could not complete ChatGPT sign-in. Try again.")
        finally:
            with self._browser_lock:
                self._notice = notice
                self._pending = False

    def _access_token(self, state: dict) -> str:
        profile = self._active_profile(state)
        if profile.get("status") == "disconnected":
            raise _sign_in_required()
        if not profile.get("access_token"):
            raise _sign_in_required()
        if SHARING_SCOPE not in profile.get("scopes", []):
            raise _permission_required()
        now = time.time()
        if profile["expires_at"] > now + 60 and not profile.get("refresh_needed"):
            return profile["access_token"]
        if now < profile.get("refresh_at", 0):
            if profile["expires_at"] > now and not profile.get("refresh_needed"):
                return profile["access_token"]
            raise PlanError("refresh_not_ready", "OpenAI has not opened the ChatGPT token renewal window. Retry shortly.")
        try:
            if not profile.get("refresh_token"):
                raise _sign_in_required()
            refreshed = self._exchange(
                client_id=profile["client_id"], grant_type="refresh_token",
                refresh_token=profile["refresh_token"],
            )
            tokens = self._tokens(refreshed, previous_scopes=profile["scopes"])
        except PlanError as exc:
            if exc.code in _REFRESH_FAILURES | {"reauth_required", "invalid_token_response"}:
                _clear_tokens(profile, "reauth_required")
                self.save(state)
                raise _sign_in_required() from None
            if exc.code == "invalid_client":
                raise PlanError("invalid_client", "OpenAI rejected this ChatGPT registration. Add a ChatGPT account again.") from None
            raise
        profile.update(tokens)
        self.save(state)
        if SHARING_SCOPE not in profile["scopes"]:
            raise _permission_required()
        return profile["access_token"]

    def understanding_connection(self) -> str:
        """Bind queued feature work to this sign-in across token refreshes."""
        with self.locked_state() as state:
            profile = self._active_profile(state)
            if not profile.get("access_token"):
                raise _sign_in_required()
            if SHARING_SCOPE not in profile.get("scopes", []):
                raise _permission_required()
            connection_id = profile.get("understanding_connection_id")
            if not connection_id:
                connection_id = uuid.uuid4().hex
                profile["understanding_connection_id"] = connection_id
                self.save(state)
            return connection_id

    def access_token(self, *, understanding_connection_id: str | None = None) -> str:
        with self.locked_state() as state:
            if understanding_connection_id is not None and (
                self._active_profile(state).get("understanding_connection_id") != understanding_connection_id
            ):
                raise PlanError("connection_changed", "The ChatGPT connection changed. Start the analysis again.")
            return self._access_token(state)

    def record_auth_failure(self, token: str) -> None:
        with self.locked_state() as state:
            profile = self._active_profile(state)
            if profile.get("access_token") == token:
                profile["refresh_needed"] = True
                self.save(state)

    def models(self) -> list[dict[str, Any]]:
        token = self.access_token()
        response = self._request("GET", f"{RESOURCE}/models", headers={"Authorization": f"Bearer {token}"})
        if not response.is_success:
            raise self.inference_error(token, response.text, status=response.status_code)
        try:
            models = response.json()["models"]
            catalog = []
            for model in models:
                if model.get("visibility") != "list" or not isinstance(model.get("slug"), str):
                    continue
                efforts = [
                    entry["effort"]
                    for entry in model.get("supported_reasoning_levels") or []
                    if entry.get("effort") in REASONING_EFFORTS
                ]
                default_effort = model.get("default_reasoning_level")
                if default_effort not in efforts:
                    default_effort = efforts[0] if efforts else ""
                if not efforts:
                    continue
                catalog.append({
                    "id": model["slug"], "label": model.get("display_name") or model["slug"],
                    "reasoningEfforts": efforts, "defaultReasoningEffort": default_effort,
                    "supportsImages": "image" in (model.get("input_modalities") or []),
                })
            return catalog
        except (ValueError, TypeError, KeyError, AttributeError):
            raise PlanError("invalid_models", "OpenAI did not return a valid ChatGPT plan model catalog.") from None

    def inference_error(
        self, token: str, raw: str, *, status: int | None = None, operation: str = "Director",
    ) -> PlanError:
        detail = raw.casefold()
        # Codex 0.164 forwards the message alone for unknown response.failed codes.
        if "subscription_sharing_usage_limit_exceeded" in detail or (
            "usage limit" in detail and ("chatgpt" in detail or "plan" in detail)
        ):
            return PlanError("usage_limit", f"DramaClaw reached a ChatGPT plan usage limit. Review limits and credits at {USAGE_URL}.")
        if "subscription_sharing_user_not_eligible" in raw:
            return PlanError("not_eligible", "ChatGPT plan usage is unavailable for this account or workspace.")
        if status == 401 or "subscription_sharing_invalid_user" in raw or "401 Unauthorized" in raw:
            self.record_auth_failure(token)
            return PlanError("session_rejected", "OpenAI rejected this ChatGPT session. Retry to renew it, or sign in again in Agent model settings.")
        if "subscription_sharing_unsupported_capability" in raw:
            return PlanError("unsupported_capability", f"ChatGPT plan rejected a capability requested by {operation}. Check the selected model and inputs.")
        if status == 403 or "403 Forbidden" in raw:
            return PlanError("permission_denied", "OpenAI denied ChatGPT plan access. Check account, workspace, and region access.")
        if status == 429 or "429 Too Many Requests" in raw:
            return PlanError("rate_limit", f"OpenAI rate-limited ChatGPT plan usage. Review usage at {USAGE_URL} or retry later.")
        if status == 503 or "subscription_sharing_usage_unavailable" in raw or "503 Service Unavailable" in raw:
            return PlanError("usage_unavailable", "ChatGPT plan usage is temporarily unavailable. Retry later.")
        if "model_not_found" in raw:
            return PlanError("model_unavailable", "The selected ChatGPT plan model is unavailable. Choose another model in Agent model settings.")
        return PlanError(
            f"{operation.lower()}_request_failed",
            f"{operation} could not complete the ChatGPT plan request. Check your model and connection, then retry.",
        )

    def disconnect(self, stop_runtime) -> None:
        with self._browser_lock:
            if self._pending:
                raise PlanError("connection_busy", "Wait for browser sign-in to finish before signing out.")
        with self.locked_state() as state:
            stop_runtime()
            profile = self._active_profile(state)
            refresh_token = profile.get("refresh_token")
            confirmed = True
            if refresh_token:
                confirmed = False
                for delay in (0, 0.5):
                    time.sleep(delay)
                    try:
                        endpoint = _auth_endpoint(self._metadata().get("revocation_endpoint"))
                        response = self._request("POST", endpoint, data={
                            "token": refresh_token, "token_type_hint": "refresh_token",
                            "client_id": profile["client_id"],
                        })
                        confirmed = response.status_code == 200
                        if response.status_code < 500:
                            break
                    except PlanError:
                        pass
            if profile:
                _clear_tokens(profile, "disconnected")
                self.save(state)
        with self._browser_lock:
            self._notice = None if confirmed else PlanError(
                "revocation_unconfirmed",
                "Signed out in DramaClaw. OpenAI revocation could not be confirmed. Disconnect DramaClaw in ChatGPT Settings.",
            )


def _storage_dir() -> Path:
    return Path.home() / ".config" / "dramaclaw" / "director-chatgpt"


SESSION = PlanSession(_storage_dir())


def available() -> bool:
    from novelvideo.chat.service import _configured_chat_backend, is_codex_backend_available
    from novelvideo.shared.runtime_env import is_ce_effective

    return (
        os.name == "posix"
        and os.environ.get("DRAMACLAW_LOCAL_CHATGPT_PLAN") == "1"
        and os.environ.get("WEB_CONCURRENCY", "1") in {"", "1"}
        and is_ce_effective()
        and _configured_chat_backend() == "codex"
        and is_codex_backend_available()
    )
