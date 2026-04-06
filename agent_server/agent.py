import logging
import os
import shutil
from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

import mlflow
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.settings import settings

logger = logging.getLogger(__name__)


def _get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)


def _get_databricks_token() -> str:
    """Get Databricks auth token from env or SDK."""
    token = os.getenv("DATABRICKS_TOKEN", "")
    if token:
        return token
    try:
        from databricks.sdk.core import Config

        cfg = Config(profile="DEFAULT")
        result = cfg.authenticate()
        headers = result() if callable(result) else result
        return headers.get("Authorization", "").removeprefix("Bearer ")
    except Exception as e:
        logger.warning("Could not get Databricks token: %s", e)
        return ""


# --- Custom tools ---


@tool("get_current_time", "Get the current date and time", {})
async def get_current_time(args):
    return {"content": [{"type": "text", "text": datetime.now().isoformat()}]}


custom_tools_server = create_sdk_mcp_server(
    "custom-tools", tools=[get_current_time]
)


def _build_agent_options() -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions configured for Databricks AI Gateway."""
    token = _get_databricks_token()

    env_vars = {
        "ANTHROPIC_BASE_URL": settings.ai_gateway_url,
        "ANTHROPIC_API_KEY": "dummy-key-for-cli-auth-check",
        "ANTHROPIC_AUTH_TOKEN": token,
        "CLAUDE_CODE_SKIP_AUTH_LOGIN": "true",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "DATABRICKS_TOKEN": token,
        "DATABRICKS_CONFIG_PROFILE": "",
        "DATABRICKS_CLIENT_ID": "",
        "DATABRICKS_CLIENT_SECRET": "",
    }

    def _log_stderr(line: str) -> None:
        logger.debug("[claude-sdk] %s", line.rstrip())

    return ClaudeAgentOptions(
        model=settings.model,
        system_prompt=settings.system_prompt,
        env=env_vars,
        mcp_servers={"custom-tools": custom_tools_server},
        allowed_tools=["Bash", "mcp__custom-tools__*"],
        max_turns=settings.max_turns,
        permission_mode="bypassPermissions",
        include_partial_messages=True,
        cli_path=shutil.which("claude"),
        stderr=_log_stderr,
    )


def _convert_request_to_prompt(request: ResponsesAgentRequest) -> str:
    """Convert Responses API input to a prompt string for the Claude Agent SDK."""
    parts = []
    for item in request.input:
        dumped = item.model_dump()
        role = dumped.get("role", "user")
        content = dumped.get("content", "")
        if isinstance(content, list):
            text_parts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "input_text"
            ]
            content = "\n".join(text_parts)
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts) if parts else ""


def _messages_to_response(messages: list) -> ResponsesAgentResponse:
    """Convert Claude SDK messages to ResponsesAgentResponse."""
    output = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    output.append(
                        {
                            "type": "message",
                            "id": str(uuid4()),
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": block.text}],
                        }
                    )
                elif isinstance(block, ToolUseBlock):
                    output.append(
                        {
                            "type": "function_call",
                            "id": str(uuid4()),
                            "call_id": getattr(block, "id", str(uuid4())),
                            "name": block.name,
                            "arguments": str(block.input),
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    content_text = (
                        block.content
                        if isinstance(block.content, str)
                        else str(block.content)
                    )
                    output.append(
                        {
                            "type": "function_call_output",
                            "id": str(uuid4()),
                            "call_id": getattr(block, "tool_use_id", str(uuid4())),
                            "output": content_text,
                        }
                    )
    return ResponsesAgentResponse(output=output)


async def _stream_to_events(
    message_iter,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Convert Claude SDK message stream to Responses API SSE events.

    - Text: streamed token-by-token via StreamEvent (content_block_start/delta/stop)
    - Tool use/result: emitted from AssistantMessage snapshots, deduplicated by ID
    - AssistantMessage TextBlocks are skipped (already handled by StreamEvents)
    """
    current_text_item_id: str | None = None
    accumulated_text = ""
    emitted_tool_ids: set[str] = set()

    async for msg in message_iter:
        if isinstance(msg, StreamEvent):
            evt = msg.event
            evt_type = evt.get("type", "")

            if evt_type == "content_block_start":
                block = evt.get("content_block", {})
                if block.get("type") == "text":
                    current_text_item_id = str(uuid4())
                    accumulated_text = ""
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.added",
                        item={
                            "type": "message",
                            "id": current_text_item_id,
                            "role": "assistant",
                            "content": [],
                        },
                    )

            elif evt_type == "content_block_delta":
                delta = evt.get("delta", {})
                if delta.get("type") == "text_delta" and current_text_item_id:
                    text = delta.get("text", "")
                    accumulated_text += text
                    yield ResponsesAgentStreamEvent(
                        type="response.output_text.delta",
                        item_id=current_text_item_id,
                        content_index=0,
                        delta=text,
                    )

            elif evt_type == "content_block_stop":
                if current_text_item_id:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item={
                            "type": "message",
                            "id": current_text_item_id,
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": accumulated_text}
                            ],
                        },
                    )
                    current_text_item_id = None
                    accumulated_text = ""

        elif isinstance(msg, AssistantMessage):
            # Only handle tool blocks — TextBlocks are already streamed via StreamEvents.
            # Deduplicate by tool ID since AssistantMessage snapshots repeat blocks.
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    block_id = getattr(block, "id", None)
                    if block_id and block_id in emitted_tool_ids:
                        continue
                    if block_id:
                        emitted_tool_ids.add(block_id)
                    item_id = str(uuid4())
                    call_id = block_id or str(uuid4())
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.added",
                        item={
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": block.name,
                            "arguments": str(block.input),
                        },
                    )
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item={
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": block.name,
                            "arguments": str(block.input),
                        },
                    )
                elif isinstance(block, ToolResultBlock):
                    block_id = getattr(block, "tool_use_id", None)
                    result_key = f"result_{block_id}"
                    if result_key in emitted_tool_ids:
                        continue
                    if result_key:
                        emitted_tool_ids.add(result_key)
                    item_id = str(uuid4())
                    content_text = (
                        block.content
                        if isinstance(block.content, str)
                        else str(block.content)
                    )
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.added",
                        item={
                            "type": "function_call_output",
                            "id": item_id,
                            "call_id": block_id or str(uuid4()),
                            "output": content_text,
                        },
                    )
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item={
                            "type": "function_call_output",
                            "id": item_id,
                            "call_id": block_id or str(uuid4()),
                            "output": content_text,
                        },
                    )

        elif isinstance(msg, ResultMessage):
            yield ResponsesAgentStreamEvent(
                type="response.completed",
                response={"status": "completed", "output": []},
            )


# --- Invoke handler ---


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := _get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    prompt = _convert_request_to_prompt(request)
    options = _build_agent_options()

    messages = []
    async for msg in query(prompt=prompt, options=options):
        messages.append(msg)

    return _messages_to_response(messages)


# --- Stream handler ---


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := _get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    prompt = _convert_request_to_prompt(request)
    options = _build_agent_options()

    async for event in _stream_to_events(query(prompt=prompt, options=options)):
        yield event
