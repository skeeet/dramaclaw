# SPDX-License-Identifier: Elastic-2.0
"""Opt-in ChatGPT plan routing for selected local understanding operations."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import asdict, dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from starlette.requests import HTTPConnection

from novelvideo.chat import chatgpt_plan

STYLE_ANALYSIS = "style_analysis"
IMAGE_REVERSE_PROMPT = "image_reverse_prompt"
SCENE_HEADINGS = "scene_headings"
IMAGE_FEATURES = frozenset({STYLE_ANALYSIS, IMAGE_REVERSE_PROMPT})
FEATURES = IMAGE_FEATURES | {SCENE_HEADINGS}

_BILLING_FEATURES = {
    "mainline.style_analysis": STYLE_ANALYSIS,
    "freezone.image_reverse_prompt": IMAGE_REVERSE_PROMPT,
}


@dataclass(frozen=True)
class PlanModel:
    model: str
    effort: str = ""
    connection_id: str | None = None

    def task_payload(self) -> dict[str, str | None]:
        return asdict(self)

    @classmethod
    def from_task(cls, value: dict | None) -> PlanModel | None:
        if value is None:
            return None
        connection_id = value.get("connection_id")
        if not isinstance(connection_id, str) or not connection_id:
            raise chatgpt_plan.PlanError("connection_changed", "Start the analysis with an active ChatGPT connection.")
        return cls(
            model=value["model"], effort=value.get("effort", ""),
            connection_id=connection_id,
        )


async def plan_for_request(feature: str, request: HTTPConnection | None) -> PlanModel | None:
    from fastapi import HTTPException

    from novelvideo.model_gateway_settings import get_understanding_plan_config

    config = get_understanding_plan_config()
    if feature not in config["features"]:
        return None
    if request is None or not chatgpt_plan.available() or not chatgpt_plan.is_local_connection(request):
        raise HTTPException(403, "ChatGPT plan understanding requires a local DramaClaw connection.")
    try:
        connection_id = await asyncio.to_thread(chatgpt_plan.SESSION.understanding_connection)
    except chatgpt_plan.PlanError as exc:
        raise HTTPException(409, str(exc)) from None
    return PlanModel(config["model"], config["reasoningEffort"], connection_id)


def uses_plan_billing(feature: str) -> bool:
    from novelvideo.model_gateway_settings import get_understanding_plan_config

    return _BILLING_FEATURES.get(feature) in get_understanding_plan_config()["features"]


@dataclass(eq=False)
class _ActiveResponse:
    loop: asyncio.AbstractEventLoop
    task: asyncio.Task[str]
    signed_out: bool = False


_active_responses: set[_ActiveResponse] = set()
_active_responses_lock = threading.Lock()


def stop_understanding_responses() -> None:
    with _active_responses_lock:
        for response in _active_responses:
            response.signed_out = True
            if not response.loop.is_closed():
                response.loop.call_soon_threadsafe(response.task.cancel)


async def _sse_events(response: httpx.Response):
    fields = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            fields.append(line.removeprefix("data:").lstrip(" "))
        elif not line and fields:
            yield json.loads("\n".join(fields))
            fields.clear()


async def respond(
    plan: PlanModel,
    prompt: str,
    *,
    instructions: str = "",
    images: list[tuple[bytes, str]] | None = None,
    timeout: float = 120,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Consume one official Responses stream using the app's OAuth grant."""
    from novelvideo.egress_context import ambient_organization_egress_context

    if not chatgpt_plan.available() or ambient_organization_egress_context() is not None:
        raise chatgpt_plan.PlanError("unavailable", "ChatGPT plan understanding requires local DramaClaw.")
    if not plan.model or (plan.effort and plan.effort not in chatgpt_plan.REASONING_EFFORTS):
        raise chatgpt_plan.PlanError("model_required", "Choose a ChatGPT plan model in Understanding model settings.")

    content = [{"type": "input_text", "text": prompt}]
    for data, media_type in images or []:
        encoded = base64.b64encode(data).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:{media_type};base64,{encoded}"})
    body = {
        "model": plan.model,
        "input": [{"role": "user", "content": content}],
        "store": False,
        "stream": True,
        **({"instructions": instructions} if instructions else {}),
        **({"reasoning": {"effort": plan.effort}} if plan.effort else {}),
    }
    task = asyncio.create_task(_stream_response(plan, body, timeout, transport))
    active = _ActiveResponse(asyncio.get_running_loop(), task)
    with _active_responses_lock:
        _active_responses.add(active)
    try:
        return await task
    except asyncio.CancelledError:
        if active.signed_out and not asyncio.current_task().cancelling():
            raise chatgpt_plan.PlanError("signed_out", "ChatGPT was signed out during analysis.") from None
        raise
    finally:
        with _active_responses_lock:
            _active_responses.discard(active)


async def _stream_response(
    plan: PlanModel, body: dict, timeout: float, transport: httpx.AsyncBaseTransport | None,
) -> str:
    session = chatgpt_plan.SESSION
    token = await asyncio.to_thread(
        session.access_token, understanding_connection_id=plan.connection_id,
    )
    deltas: list[str] = []
    length = 0
    try:
        async with asyncio.timeout(timeout), httpx.AsyncClient(
            timeout=timeout, trust_env=False, transport=transport,
        ) as client:
            async with client.stream(
                "POST", f"{chatgpt_plan.RESOURCE}/responses",
                headers={"Authorization": f"Bearer {token}"}, json=body,
            ) as response:
                if response.is_error:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise await asyncio.to_thread(
                        session.inference_error, token, detail,
                        status=response.status_code, operation="Understanding",
                    )
                async for event in _sse_events(response):
                    kind = event.get("type")
                    if kind == "response.output_text.delta":
                        chunk = event["delta"]
                        length += len(chunk)
                        if length > 1_000_000:
                            raise chatgpt_plan.PlanError("output_too_long", "ChatGPT returned too much analysis text.")
                        deltas.append(chunk)
                    elif kind in {"response.failed", "error"}:
                        detail = json.dumps(event)
                        raise await asyncio.to_thread(
                            session.inference_error, token, detail, operation="Understanding",
                        )
                    elif kind == "response.incomplete":
                        break
                    elif kind == "response.completed":
                        text = "".join(deltas).strip()
                        if text:
                            return text
                        raise chatgpt_plan.PlanError("empty_response", "ChatGPT returned no analysis text.")
    except (httpx.HTTPError, TimeoutError):
        raise chatgpt_plan.PlanError("connection_failed", "ChatGPT understanding could not reach OpenAI or timed out.") from None
    except (KeyError, ValueError, TypeError, AttributeError):
        raise chatgpt_plan.PlanError("invalid_response", "OpenAI returned an invalid understanding response.") from None
    raise chatgpt_plan.PlanError("incomplete_response", "ChatGPT understanding ended before the response completed.")


_Result = TypeVar("_Result", bound=BaseModel)


async def structured_response(
    plan: PlanModel,
    prompt: str,
    output_model: type[_Result],
    *,
    instructions: str,
    images: list[tuple[bytes, str]] | None = None,
) -> _Result:
    schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
    text = await respond(
        plan, prompt, images=images,
        instructions=f"{instructions}\nReturn only a JSON object matching this schema:\n{schema}",
    )
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2][:-3].strip()
    try:
        return output_model.model_validate_json(text)
    except ValidationError:
        raise chatgpt_plan.PlanError("invalid_output", "ChatGPT returned analysis that did not match the expected format.") from None
