"""LLM client responsible for parsing PDF pages into structured rows."""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

from openai import OpenAI
from openai.types import Response

from .costs import UsageTracker
from .models import PageExtraction


LOGGER = logging.getLogger(__name__)


class LLMTableExtractor:
    """Wraps the OpenAI Responses API to enforce a strict schema."""

    def __init__(
        self,
        model: str = "5o-mini",
        client: Optional[OpenAI] = None,
        usage_tracker: Optional[UsageTracker] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.client = client or OpenAI()
        self.usage_tracker = usage_tracker
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def extract_page(
        self, page_text: str, pending_description: Optional[str] = None
    ) -> PageExtraction:
        """Send the page text to the LLM and parse the structured response."""

        response = self._call_with_retries(
            messages=self._build_messages(page_text, pending_description)
        )
        self._record_usage(response)
        parsed_payload = self._parse_response(response)
        return PageExtraction.model_validate(parsed_payload)

    # ------------------------------------------------------------------
    def _build_messages(
        self, page_text: str, pending_description: Optional[str]
    ) -> List[dict]:
        continuation_note = (
            pending_description.strip() if pending_description and pending_description.strip() else ""
        )
        user_prompt = (
            "Analizza la trascrizione testuale di una pagina di un capitolato in PDF. "
            "La tabella contiene colonne equivalenti a codice, descrizione, quantita', prezzo unitario e totale. "
            "Restituisci esclusivamente uno schema JSON valido che rispetti il modello fornito. "
            "Se una descrizione era spezzata nella pagina precedente, la stringa fornita nel campo "
            "'pending_description' deve essere concatenata (con uno spazio) all'inizio della prima descrizione della lista. "
            "Quando la pagina termina nel mezzo di una descrizione, copia la parte finale nel campo continuation.pending_description "
            "in modo che possa essere usata come prefisso nella pagina successiva. Non duplicare testo gia' utilizzato."
        )
        content = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a meticulous data extraction engine. "
                            "Always output strict JSON that matches the provided schema."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "text",
                        "text": (
                            "Testo della pagina:\n<<<\n{page}\n>>>\n\n"
                            "pending_description precedente: {pending}"
                        ).format(
                            page=page_text.strip(),
                            pending=continuation_note or "<none>",
                        ),
                    },
                    {
                        "type": "text",
                        "text": (
                            "Schema atteso: {schema}"
                        ).format(schema=json.dumps(PageExtraction.model_json_schema())),
                    },
                ],
            },
        ]
        return content

    def _call_with_retries(self, messages: List[dict]) -> Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.responses.create(
                    model=self.model,
                    input=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "page_extraction",
                            "schema": PageExtraction.model_json_schema(),
                        },
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = exc
                if attempt == self.max_retries:
                    raise
                sleep_time = self.retry_delay * attempt
                LOGGER.warning(
                    "LLM call failed (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
        # Should never happen because of the raise above.
        raise RuntimeError(f"Failed to call the LLM: {last_error}")

    def _parse_response(self, response: Response) -> dict:
        """Extract the parsed JSON payload from the response."""

        for output in response.output or []:
            for content in output.content or []:
                if hasattr(content, "json_schema") and content.json_schema is not None:
                    return content.json_schema
                if hasattr(content, "text") and content.text:
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError as exc:  # pragma: no cover - safety
                        raise ValueError(
                            "LLM returned invalid JSON despite schema enforcement"
                        ) from exc
        raise ValueError("The LLM response did not contain any JSON payload.")

    def _record_usage(self, response: Response) -> None:
        if not self.usage_tracker:
            return
        self.usage_tracker.add_usage(self.model, response.usage)
