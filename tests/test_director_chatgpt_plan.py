from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from novelvideo.chat import chatgpt_plan as plan


def test_authorize_url_uses_dynamic_client_and_pkce():
    redirect = "http://127.0.0.1:12345/auth/callback"
    attempt = plan.Authorization(redirect)
    params = parse_qs(urlsplit(attempt.url("urn:uuid:host", None)).query)

    assert params["redirect_uri"] == [redirect]
    assert params["client_id"] == ["dynamic_agent_client"]
    assert params["agent_name_hint"] == [plan.APP_NAME]
    assert params["resource"] == [plan.RESOURCE]
    assert params["code_challenge_method"] == ["S256"]
    assert plan.SHARING_SCOPE in params["scope"][0].split()
    assert attempt.verifier not in attempt.url("urn:uuid:host", None)
    assert attempt.callback({"state": [attempt.state], "code": ["code"], "client_id": ["oaiapp_id"]}) == (
        "code", "oaiapp_id"
    )
    with pytest.raises(plan.PlanError):
        attempt.callback({"state": ["wrong"], "code": ["code"]})


def test_returning_sign_in_reuses_registration():
    attempt = plan.Authorization("http://127.0.0.1:12345/auth/callback", "oaiapp_saved")
    params = parse_qs(urlsplit(attempt.url("urn:uuid:host", {"email": "user@example.com"})).query)
    assert params["client_id"] == ["oaiapp_saved"]
    assert "agent_name_hint" not in params
    assert attempt.callback({"state": [attempt.state], "code": ["code"]}) == ("code", "oaiapp_saved")
    with pytest.raises(plan.PlanError):
        attempt.callback({"state": [attempt.state], "code": ["code"], "client_id": ["different"]})


def test_refresh_rotates_tokens_and_revocation_keeps_registration(tmp_path):
    requests = []

    def handler(request):
        requests.append(parse_qs(request.content.decode()))
        if len(requests) == 1:
            return httpx.Response(200, json={
                "access_token": "replacement", "refresh_token": "rotated",
                "expires_in": 3600, "token_type": "Bearer", "scope": plan.SCOPES,
            })
        return httpx.Response(400, json={"error": "invalid_grant"})

    session = plan.PlanSession(tmp_path, transport=httpx.MockTransport(handler))
    with session.locked_state() as state:
        state["active"] = "profile"
        state["profiles"]["profile"] = {
            "client_id": "oaiapp_test", "id_token": "identity",
            "access_token": "expired", "refresh_token": "original",
            "expires_at": 0, "scopes": plan.SCOPES.split(),
        }
        session.save(state)
    assert session.access_token() == "replacement"
    assert requests[0] == {
        "client_id": ["oaiapp_test"], "grant_type": ["refresh_token"],
        "refresh_token": ["original"], "resource": [plan.RESOURCE],
    }
    assert session.access_token() == "replacement"
    with session.locked_state() as state:
        state["profiles"]["profile"]["expires_at"] = 0
        session.save(state)
    with pytest.raises(plan.PlanError, match="Sign in"):
        session.access_token()
    assert requests[1]["refresh_token"] == ["rotated"]
    assert session.status()["status"] == "reauth_required"
    assert "original" not in session.path.read_text()
    assert "rotated" not in session.path.read_text()
    assert session.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_plan_director_streams_and_resumes_with_rotated_token(tmp_path, monkeypatch):
    import asyncio
    import os
    import time
    from pathlib import Path

    from novelvideo import config
    from novelvideo import model_gateway_settings as gateway
    from novelvideo.chat import backend_sdk, service
    from test_codex_app_server import (
        _recording_responses_gateway, _response_completed, _response_created, _responses_sse,
    )
    from test_codex_resume_refreshes_tool_runtime import _fake_project_api, _tool_call, _write_token

    binary = os.environ.get("CODEX_BIN")
    if not binary:
        pytest.skip("CODEX_BIN must point to the DramaClaw-patched runtime")

    def reply(tag, text):
        message = f"message-{tag}"
        return _responses_sse(
            _response_created(tag),
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "message", "role": "assistant", "id": message, "content": []}},
            {"type": "response.output_text.delta", "item_id": message,
             "output_index": 0, "content_index": 0, "delta": text},
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "role": "assistant", "id": message,
                         "content": [{"type": "output_text", "text": text}]},
            },
            _response_completed(tag),
        )

    def refresh(request):
        assert parse_qs(request.content.decode())["refresh_token"] == ["test-renew"]
        return httpx.Response(200, json={
            "access_token": "test-plan-rotated", "refresh_token": "test-renew-2",
            "expires_in": 3600, "token_type": "Bearer", "scope": plan.SCOPES,
        })

    credentials = plan.PlanSession(tmp_path / "private", transport=httpx.MockTransport(refresh))
    monkeypatch.setattr(plan, "SESSION", credentials)
    with credentials.locked_state() as state:
        state["active"] = "profile"
        state["profiles"]["profile"] = {
            "status": "connected", "client_id": "oaiapp_test", "access_token": "test-plan-original",
            "refresh_token": "test-renew", "expires_at": time.time() + 3600,
            "scopes": plan.SCOPES.split(),
        }
        credentials.save(state)

    monkeypatch.setenv("ST_EDITION", "ce")
    monkeypatch.setenv("DRAMACLAW_CHAT_BACKEND", "codex")
    monkeypatch.setenv("DRAMACLAW_LOCAL_CHATGPT_PLAN", "1")
    monkeypatch.setenv("NOVELVIDEO_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("NOVELVIDEO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("NOVELVIDEO_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setenv("DRAMACLAW_ROOT", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("ACCESS_TOKEN", "ambient-token")
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(gateway, "get_director_chatgpt_plan_config", lambda: ("gpt-6.2-sol", "low"))

    def unexpected_gateway(*_):
        pytest.fail("A ChatGPT plan turn resolved the existing gateway credential")

    monkeypatch.setattr(service, "_codex_turn_gateway_credentials", unexpected_gateway)

    turn_token = tmp_path / "token"
    _write_token(turn_token, "test-turn-one")
    responses = [
        _tool_call(1, "mcp__dramaclaw", "dramaclaw_list_tasks"),
        reply("final", "No pending tasks."),
        reply("resumed", "History retained."),
        _responses_sse(
            _response_created("limited"),
            {
                "type": "response.failed",
                "response": {"id": "limited", "error": {
                    "type": "rate_limit_error",
                    "code": "subscription_sharing_usage_limit_exceeded",
                    "message": "Plan usage limit reached.",
                }},
            },
        ),
    ]
    with _fake_project_api("project") as (app_url, received):
        monkeypatch.setattr(service, "_load_api_url", lambda: app_url)
        with _recording_responses_gateway(responses) as (model_url, requests):
            monkeypatch.setattr(plan, "RESOURCE", model_url)
            options = {"project_state_dir": tmp_path / "project", "agent_token_file": turn_token}
            first = service._build_codex_thread("local", "test-project", "", **options)
            assert "ACCESS_TOKEN" not in first._env
            assert not first._turn_metadata
            async def follow(thread_id, turn_id):
                return [event async for event in backend_sdk.follow_live_plan_turn(thread_id, turn_id)]

            stream = first.stream("Use the read-only list tasks tool.")
            started = await anext(stream)
            follower = asyncio.create_task(follow(started.thread_id, started.turn_id))
            events = [event async for event in stream]
            calls = [event for event in events if event.type == "tool_updated"]
            assert received == ["first"]
            assert calls[0].structured["ok"] is True and calls[0].status == "completed"
            assert any(event.text == "No pending tasks." for event in events if event.type == "assistant_delta")
            assert any(event.type == "complete" for event in events)
            assert any(event.type == "tool_updated" for event in await follower)
            service._set_codex_thread_id("local", "test-project", first.id, project_state_dir=tmp_path / "project")
            with credentials.locked_state() as state:
                state["profiles"]["profile"]["expires_at"] = 0
                credentials.save(state)

            monkeypatch.setattr(gateway, "get_director_chatgpt_plan_config", lambda: ("gpt-6.2-sol", "max"))
            second = service._build_codex_thread("local", "test-project", "", **options)
            next_events = [event async for event in second.stream("Confirm the previous result.")]
            assert second.id == first.id
            assert any(event.text == "History retained." for event in next_events if event.type == "complete")
            limited = service._build_codex_thread("local", "test-project", "", **options)
            with pytest.raises(plan.PlanError, match="usage limit") as error:
                _ = [event async for event in limited.stream("This request is usage limited.")]
            assert error.value.code == "usage_limit"

    assert [
        {key.lower(): value for key, value in item["headers"].items()}["authorization"]
        for item in requests
    ] == ["Bearer test-plan-original", "Bearer test-plan-original", "Bearer test-plan-rotated", "Bearer test-plan-rotated"]
    assert all(request["path"] == "/v1/responses" for request in requests)
    assert all(request["body"]["store"] is False and request["body"]["stream"] is True for request in requests)
    assert all(
        "test-plan-" not in str(request["body"]) and "ambient-" not in str(request["body"])
        for request in requests
    )
    assert any(item["type"] == "function_call_output" for item in requests[1]["body"]["input"])
    assert [item["body"]["reasoning"]["effort"] for item in requests] == ["low", "low", "max", "max"]
    assert all(
        tool["type"] != "tool_search" for request in requests for tool in request["body"]["tools"]
    )


def test_plan_model_catalog_lists_supported_efforts(tmp_path, monkeypatch):
    def catalog(request):
        assert request.headers["Authorization"] == "Bearer test-model-token"
        return httpx.Response(200, json={"models": [
            {"slug": "hidden", "visibility": "hidden"},
            {
                "slug": "gpt-6.2-sol", "display_name": "GPT-6.2-Sol", "visibility": "list",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low"}, {"effort": "medium"}, {"effort": "max"}, {"effort": "ultra"},
                ],
            },
        ]})

    session = plan.PlanSession(tmp_path, transport=httpx.MockTransport(catalog))
    monkeypatch.setattr(session, "access_token", lambda: "test-model-token")
    assert session.models() == [{
        "id": "gpt-6.2-sol", "label": "GPT-6.2-Sol", "reasoningEfforts": ["low", "medium", "max"],
        "defaultReasoningEffort": "medium",
        "supportsImages": False,
    }]


@pytest.mark.parametrize(
    ("host", "peer", "origin", "allowed"),
    [
        ("127.0.0.1:8780", "127.0.0.1", "http://127.0.0.1:8780", True),
        ("localhost:5173", "::ffff:127.0.0.1", None, True),
        ("127.0.0.1:8780", "192.0.2.8", None, False),
        ("127.0.0.1:8780", "127.0.0.1", "https://example.org", False),
        ("example.org", "127.0.0.1", None, False),
    ],
)
def test_plan_rejects_untrusted_inference_connections(monkeypatch, host, peer, origin, allowed):
    from fastapi import HTTPException, WebSocket
    from novelvideo import model_gateway_settings as gateway
    from novelvideo.api.routes.chat import _require_local_plan_inference

    monkeypatch.setattr(gateway, "get_director_chatgpt_plan_config", lambda: ("gpt-6.2-sol", "low"))
    headers = [(b"host", host.encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    async def unexpected_io(*_):
        pytest.fail("The connection check must not read or write a frame.")

    connection = WebSocket({
        "type": "websocket", "scheme": "ws", "path": "/api/v1/chat/ws", "headers": headers,
        "client": (peer, 12345), "query_string": b"",
    }, unexpected_io, unexpected_io)
    if allowed:
        _require_local_plan_inference(connection)
    else:
        with pytest.raises(HTTPException) as error:
            _require_local_plan_inference(connection)
        assert error.value.status_code == 403


def test_signout_revokes_and_retains_the_client_registration(tmp_path):
    operations = []

    def oauth(request):
        if request.method == "GET":
            return httpx.Response(200, json={
                "issuer": plan.ISSUER, "revocation_endpoint": plan.ISSUER + "/api/accounts/oauth/revoke",
            })
        assert operations == ["stopped"]
        operations.append("revoked")
        assert parse_qs(request.content.decode()) == {
            "token": ["refresh"], "token_type_hint": ["refresh_token"], "client_id": ["oaiapp_test"]
        }
        return httpx.Response(200)

    session = plan.PlanSession(tmp_path, transport=httpx.MockTransport(oauth))
    with session.locked_state() as state:
        state["active"] = "profile"
        state["profiles"]["profile"] = {
            "label": "Connection 1", "client_id": "oaiapp_test", "status": "connected",
            "id_token": "identity", "access_token": "dated", "refresh_token": "refresh",
            "scopes": plan.SCOPES.split(),
        }
        session.save(state)
    session.disconnect(lambda: operations.append("stopped"))
    assert operations == ["stopped", "revoked"]
    assert session.status()["status"] == "disconnected"
    with session.locked_state() as state:
        profile = state["profiles"]["profile"]
        assert profile["client_id"] == "oaiapp_test"
        assert {"access_token", "refresh_token", "id_token"}.isdisjoint(profile)
