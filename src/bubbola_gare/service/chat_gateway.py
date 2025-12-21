from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mcp.client.session_group import ClientSessionGroup, StreamableHttpParameters
from openai import OpenAI

from ..config import CHAT_MODEL, MCP_HTTP_URL_COMMESSE, MCP_HTTP_URL_ORDERS, SQL_DEFAULT_LIMIT
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
        *,
        mcp_url: str,
        model: str,
        allowed_tools: set[str],
        schema_tool: str,
        run_sql_tool: str,
        ask_tool: str | None,
        semantic_tool: str | None,
        dataset_label: str,
        tool_descriptions: dict[str, str],
        client: Optional[OpenAI] = None,
    ):
        self.model = model
        self.client = client or get_openai_client()
        self.mcp = MCPClientManager(mcp_url)
        self.allowed_tools = allowed_tools
        self.schema_tool = schema_tool
        self.run_sql_tool = run_sql_tool
        self.ask_tool = ask_tool
        self.semantic_tool = semantic_tool
        self.dataset_label = dataset_label
        self.tool_descriptions = tool_descriptions
        self._sql_retry_tools = {tool for tool in (run_sql_tool, ask_tool) if tool}

    @classmethod
    def for_orders(cls, model: str = CHAT_MODEL, client: Optional[OpenAI] = None) -> "ChatGateway":
        descriptions = {
            "ask_orders": "NL/analytics for the orders table; use when the intent mentions orders/ordine/importo ordine or is ambiguous.",
            "semantic_search": "Similarity over orders (text or order_code with optional filters like region/vendor/etc.).",
            "run_sql_orders": "Use only when the user provides explicit SQL to query orders.",
            "list_schema_orders": "List tables/columns for the orders dataset.",
        }
        return cls(
            mcp_url=MCP_HTTP_URL_ORDERS,
            model=model,
            allowed_tools=set(descriptions.keys()),
            schema_tool="list_schema_orders",
            run_sql_tool="run_sql_orders",
            ask_tool="ask_orders",
            semantic_tool="semantic_search",
            dataset_label="orders",
            tool_descriptions=descriptions,
            client=client,
        )

    @classmethod
    def for_commesse(cls, model: str = CHAT_MODEL, client: Optional[OpenAI] = None) -> "ChatGateway":
        descriptions = {
            "ask_commesse": "NL/analytics for the commesse/projects table; use when the intent is clearly about commesse/commessa history.",
            "semantic_search_commesse": "Similarity over commesse (text or commessa_code with filters like committente/settore/responsabili/date/importo).",
            "run_sql_commesse": "Use only when the user provides explicit SQL to query commesse.",
            "list_schema_commesse": "List tables/columns for the commesse dataset.",
        }
        return cls(
            mcp_url=MCP_HTTP_URL_COMMESSE,
            model=model,
            allowed_tools=set(descriptions.keys()),
            schema_tool="list_schema_commesse",
            run_sql_tool="run_sql_commesse",
            ask_tool="ask_commesse",
            semantic_tool="semantic_search_commesse",
            dataset_label="commesse",
            tool_descriptions=descriptions,
            client=client,
        )

    async def _plan_tool(self, question: str) -> tuple[str, dict[str, Any]]:
        tools_text = "\n".join(f"- {name}: {desc}" for name, desc in self.tool_descriptions.items())
        system = (
            f"You route user questions to MCP tools for the {self.dataset_label} dataset. "
            "Return a JSON object with keys `tool` and `arguments`. Allowed tools:\n"
            f"{tools_text}\n"
            "Routing hints (soft): stay within the current dataset; if the question is SQL, choose the SQL tool; "
            "use similarity tools only when the user explicitly wants \"simile\"/\"similar\"/\"like\" or free-text similarity."
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
        if tool not in self.allowed_tools:
            raise ValueError(f"Unsupported tool selection: {tool}")
        if tool == self.run_sql_tool and not _looks_like_sql(question):
            raise ValueError("Planner selected run_sql but input does not look like SQL.")
        return tool, arguments

    def _normalize_tool_args(self, question: str, tool: str, arguments: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        """
        Clean up planner arguments and add fallbacks so MCP tools do not error on missing inputs.
        """
        args: dict[str, Any] = dict(arguments or {})
        filters = args.pop("filters", None)
        if isinstance(filters, dict):
            args = {**args, **filters}

        # Planner may return `query`; normalize to the expected `question`.
        if tool == self.ask_tool and "question" not in args and "query" in args:
            args["question"] = args.pop("query")

        if tool == self.semantic_tool:
            alias_map = {
                "importo_min": "min_importo",
                "importo_max": "max_importo",
                "min_amount": "min_importo",
                "max_amount": "max_importo",
            }
            if self.dataset_label == "orders":
                alias_map = {
                    "importo_min": "min_amount",
                    "importo_max": "max_amount",
                    "amount_min": "min_amount",
                    "amount_max": "max_amount",
                }
            for alias, target in alias_map.items():
                if alias in args and target not in args:
                    args[target] = args.pop(alias)

        if tool == self.ask_tool and "question" not in args:
            args["question"] = question

        if tool == self.semantic_tool:
            has_anchor = any(args.get(key) for key in ("query_text", "order_code", "commessa_code"))
            if not has_anchor:
                # If the planner picked semantic search without a vector anchor, fall back to NL/SQL tool.
                if self.ask_tool:
                    return self.ask_tool, {"question": question}
                args["query_text"] = question

        return tool, args

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
        if tool == self.schema_tool:
            columns = tool_result.get("columns", {})
            tables: list[str] = list(columns.keys()) if isinstance(columns, dict) else []
            if not tables:
                schema_txt = str(tool_result.get("schema", ""))
                tables = [line.split(":")[1].strip() for line in schema_txt.splitlines() if line.lower().startswith("table:")]
            tables_clean = ", ".join(tables) if tables else "nessuna tabella trovata"
            answer = f"Schema {self.dataset_label}: {tables_clean}."
            return answer, {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        system = (
            f"You are an assistant answering questions over {self.dataset_label} data using MCP tool outputs. "
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
        schema = await self._call_mcp(self.schema_tool, {})
        schema_text = schema.get("schema") or ""
        columns = schema.get("columns") or []
        system = (
            f"You fix broken SQL for the available {self.dataset_label} tables. Only use SELECT/CTE, never modify data. "
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
        if not error_text or tool not in self._sql_retry_tools:
            return tool, tool_args, tool_result

        repaired_sql, repair_meta = await self._repair_sql_with_schema(
            question, tool_args.get("sql") if isinstance(tool_args, dict) else None, error_text
        )
        if not repaired_sql:
            merged = dict(tool_result)
            merged["repair"] = repair_meta
            return tool, tool_args, merged

        retry_args = {"sql": repaired_sql, "max_rows": tool_args.get("max_rows", SQL_DEFAULT_LIMIT)}
        retry_result = await self._call_mcp(self.run_sql_tool, retry_args)
        combined_result = {
            "initial_error": error_text,
            "retry_sql": repaired_sql,
            "retry_rationale": repair_meta.get("rationale"),
            "retry_result": retry_result,
            "schema": repair_meta.get("schema"),
        }
        return self.run_sql_tool, retry_args, combined_result

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
        tool, tool_args = self._normalize_tool_args(question, tool, tool_args)
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


# Shared singletons for FastAPI routes
chat_gateway_orders = ChatGateway.for_orders()
chat_gateway_commesse = ChatGateway.for_commesse()
