from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mcp.client.session_group import ClientSessionGroup, StreamableHttpParameters
from openai import OpenAI

from ..config import CHAT_MODEL, MCP_HTTP_URL, SQL_DEFAULT_LIMIT
from ..embedding import get_openai_client

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Maintains a single MCP connection (streamable HTTP) for reuse.
    """

    def __init__(self, url: str):
        self.url = url
        self.group = ClientSessionGroup()
        self.params = StreamableHttpParameters(url=url)
        self._connected = False
        self._lock = asyncio.Lock()

    async def ensure_connected(self) -> None:
        if self._connected:
            return
        async with self._lock:
            if self._connected:
                return
            await self.group.connect_to_server(self.params)
            self._connected = True
            logger.info("Connected MCP client to %s", self.url)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
        await self.ensure_connected()
        return await self.group.call_tool(name, arguments=arguments or {})

    async def close(self) -> None:
        if self._connected:
            await self.group.__aexit__(None, None, None)  # type: ignore[arg-type]
            self._connected = False


def _parse_tool_result(result) -> dict[str, Any]:
    """
    Normalize MCP CallToolResult into a simple dict for prompting/response.
    """
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    content_blocks = getattr(result, "content", []) or []
    texts = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    if not texts:
        return {}
    try:
        merged = "\n".join(texts)
        return json.loads(merged)
    except Exception:
        return {"text": "\n".join(texts)}


def _truncate_for_prompt(obj: Any, max_chars: int = 4000) -> str:
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3] + "..."


def _safe_token_usage(usage) -> dict[str, Optional[int]]:
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _looks_like_sql(text: str) -> bool:
    lowered = text.lower()
    return ("select" in lowered and "from" in lowered) or "with " in lowered or ";" in lowered


def _extract_error(tool_result: dict[str, Any]) -> str | None:
    if not isinstance(tool_result, dict):
        return None
    if tool_result.get("error"):
        return str(tool_result["error"])
    text = tool_result.get("text")
    if isinstance(text, str) and "error" in text.lower():
        return text
    return None


@dataclass
class ChatResult:
    answer: str
    tool_used: str
    tool_arguments: dict[str, Any]
    tool_result: dict[str, Any]
    token_usage: dict[str, Optional[int]]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_used": self.tool_used,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "token_usage": self.token_usage,
            "model": self.model,
        }


class ChatGateway:
    def __init__(
        self,
        mcp_url: str = MCP_HTTP_URL,
        model: str = CHAT_MODEL,
        client: Optional[OpenAI] = None,
    ):
        self.model = model
        self.client = client or get_openai_client()
        self.mcp = MCPClientManager(mcp_url)

    async def _plan_tool(self, question: str) -> tuple[str, dict[str, Any]]:
        system = (
            "You route user questions to MCP tools. Return a JSON object with keys "
            "`tool` and `arguments`. Allowed tools:\n"
            "- ask_orders: NL/analytics for the procurement orders table; use when the intent mentions orders/ordine/importo ordine or is ambiguous.\n"
            "- ask_commesse: NL/analytics for the commesse/projects table; use when the intent is clearly about commesse/commessa history.\n"
            "- semantic_search: similarity over orders (text or order_code with optional filters like region/vendor/etc.).\n"
            "- semantic_search_commesse: similarity over commesse (text or commessa_code with filters like committente/settore/responsabili/date/importo).\n"
            "- run_sql: when the user provides explicit SQL.\n"
            "- list_schema: when the user asks about tables/columns/schema.\n"
            "Routing hints (soft): if the question mentions commessa/commesse, consider ask_commesse; if it mentions ordine/importo ordine/order code, consider ask_orders; use semantic_search only when the user explicitly asks for \"simile\"/\"similar\"/\"like\" or free-text similarity."
        )
        user = f"Question: {question}\nRespond ONLY with JSON."
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        tool = str(parsed.get("tool", "")).strip()
        arguments = parsed.get("arguments") or {}
        if tool not in {"ask_orders", "ask_commesse", "semantic_search", "semantic_search_commesse", "run_sql", "list_schema"}:
            raise ValueError(f"Unsupported tool selection: {tool}")
        if tool == "run_sql" and not _looks_like_sql(question):
            raise ValueError("Planner selected run_sql but input does not look like SQL.")
        return tool, arguments

    async def _call_mcp(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.mcp.call_tool(tool, arguments)
        return _parse_tool_result(result)

    async def _compose_answer(
        self,
        question: str,
        tool: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> tuple[str, dict[str, Optional[int]]]:
        if tool == "list_schema":
            columns = tool_result.get("columns", {})
            tables: list[str] = list(columns.keys()) if isinstance(columns, dict) else []
            if not tables:
                schema_txt = str(tool_result.get("schema", ""))
                tables = [line.split(":")[1].strip() for line in schema_txt.splitlines() if line.lower().startswith("table:")]
            tables_clean = ", ".join(tables) if tables else "nessuna tabella trovata"
            answer = f"Tabelle disponibili: {tables_clean}."
            return answer, {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        system = (
            "You are an assistant answering questions over orders and commesse data using MCP tool outputs. "
            "Use the provided results to answer concisely. "
            "If the tool is list_schema, explicitly name the available tables before mentioning columns. "
            "If no data is available, say so. Include the SQL or search intent briefly when helpful."
        )
        tool_summary = _truncate_for_prompt({"tool": tool, "arguments": tool_args, "result": tool_result})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "system", "content": f"Tool output:\n{tool_summary}"},
        ]
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            temperature=0.2,
            messages=messages,
        )
        answer = resp.choices[0].message.content or ""
        usage = _safe_token_usage(getattr(resp, "usage", None))
        return answer, usage

    async def _repair_sql_with_schema(
        self, question: str, original_sql: str | None, error_text: str
    ) -> tuple[str | None, dict[str, Any]]:
        schema = await self._call_mcp("list_schema", {})
        schema_text = schema.get("schema") or ""
        columns = schema.get("columns") or []
        system = (
            "You fix broken SQL for the available tables (`orders`, `commesse`). Only use SELECT/CTE, never modify data. "
            f"Use the provided schema and keep LIMIT <= {SQL_DEFAULT_LIMIT}. Respond with JSON containing `sql` and optional `rationale`."
        )
        user = (
            f"Question: {question}\n"
            f"Previous SQL: {original_sql or '(none)'}\n"
            f"Error: {error_text}\n"
            f"Schema:\n{schema_text}\n"
            f"Columns: {columns}"
        )
        try:
            resp = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            parsed = json.loads(resp.choices[0].message.content or "{}")
            sql = str(parsed.get("sql", "")).strip()
            if not sql:
                return None, {"error": "empty_repair_sql", "schema": schema}
            rationale = parsed.get("rationale")
            return sql, {"rationale": rationale, "schema": schema}
        except Exception as exc:
            logger.error("SQL repair failed: %s", exc)
            return None, {"error": str(exc), "schema": schema}

    async def _retry_on_error(
        self,
        question: str,
        tool: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        error_text = _extract_error(tool_result)
        if not error_text or tool not in {"run_sql", "ask_orders"}:
            return tool, tool_args, tool_result

        repaired_sql, repair_meta = await self._repair_sql_with_schema(
            question, tool_args.get("sql") if isinstance(tool_args, dict) else None, error_text
        )
        if not repaired_sql:
            merged = dict(tool_result)
            merged["repair"] = repair_meta
            return tool, tool_args, merged

        retry_args = {"sql": repaired_sql, "max_rows": tool_args.get("max_rows", SQL_DEFAULT_LIMIT)}
        retry_result = await self._call_mcp("run_sql", retry_args)
        combined_result = {
            "initial_error": error_text,
            "retry_sql": repaired_sql,
            "retry_rationale": repair_meta.get("rationale"),
            "retry_result": retry_result,
            "schema": repair_meta.get("schema"),
        }
        return "run_sql", retry_args, combined_result

    async def chat(self, question: str) -> ChatResult:
        try:
            tool, tool_args = await self._plan_tool(question)
        except Exception as exc:
            err_msg = f"Tool planning failed: {exc}"
            logger.error(err_msg)
            return ChatResult(
                answer=err_msg,
                tool_used="planning_error",
                tool_arguments={},
                tool_result={"error": err_msg},
                token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                model=self.model,
            )
        try:
            tool_result = await self._call_mcp(tool, tool_args)
        except (Exception, BaseExceptionGroup) as exc:
            err_msg = f"MCP call failed for '{tool}' (url={self.mcp.url}): {exc}"
            logger.error(err_msg, exc_info=True)
            return ChatResult(
                answer=err_msg,
                tool_used=tool,
                tool_arguments=tool_args,
                tool_result={"error": err_msg},
                token_usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                model=self.model,
            )
        tool, tool_args, tool_result = await self._retry_on_error(question, tool, tool_args, tool_result)
        answer, usage = await self._compose_answer(question, tool, tool_args, tool_result)
        if usage.get("total_tokens") is not None:
            answer = (
                f"{answer}\n\n[token_usage: prompt={usage.get('prompt_tokens')}, "
                f"completion={usage.get('completion_tokens')}, total={usage.get('total_tokens')}]"
            )
        return ChatResult(
            answer=answer,
            tool_used=tool,
            tool_arguments=tool_args,
            tool_result=tool_result,
            token_usage=usage,
            model=self.model,
        )


# Shared singleton for FastAPI routes
chat_gateway = ChatGateway()
