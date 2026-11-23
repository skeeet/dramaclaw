import json

import httpx
import pytest

from novelvideo import model_understanding as understanding
from novelvideo.chat import chatgpt_plan


@pytest.mark.asyncio
async def test_uses_plan_bearer_and_waits_for_completed_stream(monkeypatch):
    monkeypatch.setattr(chatgpt_plan, "available", lambda: True)
    monkeypatch.setattr(chatgpt_plan.SESSION, "access_token", lambda **_: "test-siwc-token")

    def respond(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-siwc-token"
        body = json.loads(request.content)
        assert body == {
            "model": "test-model",
            "reasoning": {"effort": "low"},
            "store": False,
            "stream": True,
            "input": [{
                "role": "user", "content": [
                    {"type": "input_text", "text": "Describe the reference."},
                    {"type": "input_image", "image_url": "data:image/png;base64,YWJj"},
                ],
            }],
        }
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=(
            'data: {"type":"response.output_text.delta","delta":"Blue"}\n\n'
            'data: {"type":"response.output_text.delta","delta":" sky."}\n\n'
            'data: {"type":"response.completed"}\n\n'
        ))

    assert await understanding.respond(
        understanding.PlanModel("test-model", "low"), "Describe the reference.",
        images=[(b"abc", "image/png")], transport=httpx.MockTransport(respond),
    ) == "Blue sky."


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_partial_output_is_never_used_on_plan_failure(monkeypatch, complete):
    monkeypatch.setattr(chatgpt_plan, "available", lambda: True)
    monkeypatch.setattr(chatgpt_plan.SESSION, "access_token", lambda **_: "test-siwc-token")
    events = 'data: {"type":"response.output_text.delta","delta":"Partial text"}\n\n'
    if complete:
        events += (
            'data: {"type":"response.failed","response":{"error":'
            '{"code":"subscription_sharing_usage_limit_exceeded"}}}\n\n'
        )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=events))
    with pytest.raises(chatgpt_plan.PlanError) as error:
        await understanding.respond(understanding.PlanModel("test-model"), "Analyze this.", transport=transport)
    assert error.value.code == ("usage_limit" if complete else "incomplete_response")


async def test_switching_off_analysis_restores_channel_and_rejects_cross_origin(monkeypatch):
    from fastapi import FastAPI
    from novelvideo import model_gateway_settings
    from novelvideo.api.routes import model_gateway

    stored = {}
    monkeypatch.setattr(chatgpt_plan, "available", lambda: True)
    monkeypatch.setattr(model_gateway_settings, "get_model_gateway_settings", lambda: stored)
    monkeypatch.setattr(model_gateway_settings, "_write_many", stored.update)
    monkeypatch.setattr(chatgpt_plan.SESSION, "models", lambda: [{
        "id": "text-only", "supportsImages": False, "defaultReasoningEffort": "low",
        "reasoningEfforts": ["low"],
    }, {
        "id": "vision", "supportsImages": True, "defaultReasoningEffort": "medium",
        "reasoningEfforts": ["low", "medium"],
    }])
    app = FastAPI()
    app.include_router(model_gateway.router)
    path = "/model-gateway/agent/chatgpt-plan/understanding"
    headers = {"X-DramaClaw-Local": "1"}
    body = {"features": [understanding.IMAGE_REVERSE_PROMPT], "model": "text-only"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 23456)),
        base_url="http://127.0.0.1:8780",
    ) as client:
        invalid = await client.post(path, headers=headers, json=body)
        assert invalid.status_code == 400
        body["model"] = "vision"
        forged = await client.post(path, headers={**headers, "Origin": "https://example.com"}, json=body)
        assert forged.status_code == 403
        saved = await client.post(path, headers=headers, json=body)
        assert saved.json()["data"] == {
            "features": [understanding.IMAGE_REVERSE_PROMPT], "model": "vision",
            "reasoningEffort": "medium",
        }
        assert understanding.uses_plan_billing("freezone.image_reverse_prompt")
        assert not understanding.uses_plan_billing("mainline.style_analysis")
        monkeypatch.setattr(chatgpt_plan, "available", lambda: False)
        stopped = await client.post(path, headers=headers, json={"features": []})
        assert stopped.json()["data"]["features"] == []
        assert not understanding.uses_plan_billing("freezone.image_reverse_prompt")


async def test_scene_headings_use_plan_and_preserve_source_location(monkeypatch):
    from novelvideo.cognee import screenplay_normalizer

    def not_gateway():
        pytest.fail("Selected plan must not construct a gateway model.")

    async def analyze(plan, prompt, output_model, *, instructions):
        assert plan.model == "test-model" and "INT. Garage" in prompt
        assert instructions == screenplay_normalizer.SCREENPLAY_NORMALIZER_SYSTEM_PROMPT
        return output_model(location="Something else", time_of_day="夜")

    monkeypatch.setattr(screenplay_normalizer, "_create_screenplay_normalizer_agent", not_gateway)
    monkeypatch.setattr(understanding, "structured_response", analyze)
    result = await screenplay_normalizer.normalize_screenplay_scene_header(
        "INT. Garage", location_hint="Garage", context_lines=["A light goes out."],
        understanding_plan=understanding.PlanModel("test-model"),
    )
    assert result.location == "Garage"


async def test_image_to_prompt_uses_compact_image_without_gateway(monkeypatch, tmp_path):
    from PIL import Image
    from novelvideo.freezone import image_node

    source = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "blue").save(source)

    async def analyze(plan, prompt, *, images, timeout):
        assert plan.model == "test-model"
        assert "blue lighting" in prompt
        assert len(images) == 1 and images[0][1].startswith("image/")
        return "```\nBlue sky\n```"

    async def not_gateway(*_):
        pytest.fail("Selected plan must not dispatch image understanding through NewAPI.")

    monkeypatch.setattr(understanding, "respond", analyze)
    monkeypatch.setattr(image_node, "_reverse_prompt_with_gateway", not_gateway)
    result = await image_node.reverse_prompt_from_image(
        image_path=source, instruction="blue lighting", understanding_plan=understanding.PlanModel("test-model"),
    )
    assert result == "Blue sky"


async def test_sign_out_cancels_live_understanding(monkeypatch):
    import asyncio

    monkeypatch.setattr(chatgpt_plan, "available", lambda: True)
    monkeypatch.setattr(chatgpt_plan.SESSION, "access_token", lambda **_: "test-siwc-token")
    submitted = asyncio.Event()

    async def respond(_):
        submitted.set()
        await asyncio.Future()

    task = asyncio.create_task(understanding.respond(
        understanding.PlanModel("test-model"), "Analyze this.", transport=httpx.MockTransport(respond),
    ))
    await asyncio.wait_for(submitted.wait(), timeout=3)
    understanding.stop_understanding_responses()
    with pytest.raises(chatgpt_plan.PlanError) as error:
        await asyncio.wait_for(task, timeout=3)
    assert error.value.code == "signed_out"


async def test_style_route_uses_plan_instead_of_reserving_gateway_credits(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace

    from fastapi import UploadFile
    from starlette.requests import Request

    from novelvideo import model_gateway_settings, ports
    from novelvideo.api.routes import styles
    from novelvideo.generators.style_analyzer import StyleAnalyzer

    async def resolve(*_args, **_kwargs):
        return SimpleNamespace()

    async def analyze(data, plan):
        assert data == b"image data"
        assert plan == understanding.PlanModel("vision", "low", "connection")
        return {"style_tag": "ART STYLE"}

    def not_gateway():
        pytest.fail("Selected plan must not reserve gateway credits.")

    monkeypatch.setattr(chatgpt_plan, "available", lambda: True)
    monkeypatch.setattr(chatgpt_plan.SESSION, "understanding_connection", lambda: "connection")
    monkeypatch.setattr(model_gateway_settings, "get_understanding_plan_config", lambda: {
        "features": [understanding.STYLE_ANALYSIS], "model": "vision", "reasoningEffort": "low",
    })
    monkeypatch.setattr(styles, "resolve_project_scope", resolve)
    monkeypatch.setattr(StyleAnalyzer, "analyze_with_chatgpt_plan", analyze)
    monkeypatch.setattr(ports, "get_usage_meter", not_gateway)

    request = Request({
        "type": "http", "scheme": "http", "path": "/styles/analyze",
        "client": ("127.0.0.1", 49000),
        "headers": [(b"host", b"127.0.0.1:8780")],
    })
    upload = UploadFile(BytesIO(b"image data"), filename="reference.png")
    result = await styles.analyze_style("demo", upload, "", {}, request)
    assert result == {"ok": True, "data": {"style_tag": "ART STYLE"}}


def test_queued_plan_binding_cannot_spend_a_new_session(tmp_path):
    import time

    session = chatgpt_plan.PlanSession(tmp_path)

    def active_profile(key):
        with session.locked_state() as state:
            state["active"] = key
            state["profiles"][key] = {
                "status": "connected", "client_id": "oaiapp_test",
                "scopes": chatgpt_plan.SCOPES.split(), "access_token": "fake-" + key,
                "expires_at": time.time() + 3600,
            }
            session.save(state)

    active_profile("first")
    connection_id = session.understanding_connection()
    queued = understanding.PlanModel("vision", "low", connection_id)
    assert understanding.PlanModel.from_task(queued.task_payload()) == queued
    assert session.access_token(understanding_connection_id=queued.connection_id) == "fake-first"
    active_profile("second")
    session.understanding_connection()
    with pytest.raises(chatgpt_plan.PlanError, match="connection changed"):
        session.access_token(understanding_connection_id=queued.connection_id)
    active_profile("first")
    assert session.understanding_connection() != connection_id
    with pytest.raises(chatgpt_plan.PlanError, match="connection changed"):
        session.access_token(understanding_connection_id=queued.connection_id)
