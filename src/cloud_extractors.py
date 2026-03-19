from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, Field


class ExtractionError(RuntimeError):
    pass


class KycEntity(BaseModel):
    label: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class KycExtractionSchema(BaseModel):
    company_name: Optional[str] = None
    pan: Optional[str] = None
    gstins: list[str] = Field(default_factory=list)
    cin: Optional[str] = None
    contacts: list[str] = Field(default_factory=list)
    addresses: list[str] = Field(default_factory=list)
    entities: list[KycEntity] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class GstinExtractionSchema(BaseModel):
    company_name: Optional[str] = None
    gstins: list[str] = Field(default_factory=list)
    supporting_entities: list[KycEntity] = Field(default_factory=list)
    reconciliation: dict[str, Any] = Field(default_factory=dict)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _call_openai_json_schema(
    schema: Type[SchemaT],
    system_prompt: str,
    user_prompt: str,
) -> SchemaT:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ExtractionError("OPENAI_API_KEY is not configured.")

    payload = {
        "model": os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4o-mini"),
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        },
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    message = data["choices"][0]["message"]
    refusal = message.get("refusal")
    if refusal:
        raise ExtractionError(f"Model refusal: {refusal}")

    return schema.model_validate_json(message["content"])


def _call_groq_json(
    schema: Type[SchemaT],
    system_prompt: str,
    user_prompt: str,
) -> SchemaT:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ExtractionError("GROQ_API_KEY is not configured.")

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    payload = {
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n"
                    "Return JSON only. Do not add commentary.\n"
                    f"Match this JSON schema exactly: {schema_json}"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    return schema.model_validate(json.loads(content))


def run_structured_extraction(
    schema: Type[SchemaT],
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, SchemaT]:
    if os.getenv("OPENAI_API_KEY"):
        return "openai", _call_openai_json_schema(schema, system_prompt, user_prompt)

    if os.getenv("GROQ_API_KEY"):
        return "groq", _call_groq_json(schema, system_prompt, user_prompt)

    raise ExtractionError("Neither OPENAI_API_KEY nor GROQ_API_KEY is configured.")


def extract_kyc_entities(text: str) -> tuple[str, KycExtractionSchema]:
    provider, result = run_structured_extraction(
        KycExtractionSchema,
        (
            "Extract only the KYC entities present in the input. "
            "Do not infer missing identifiers. "
            "Populate arrays only with values explicitly supported by the text."
        ),
        text,
    )
    return provider, result


def extract_gstin_details(text: str, company_name: Optional[str]) -> tuple[str, GstinExtractionSchema]:
    company_hint = company_name or "No company hint was supplied."
    provider, result = run_structured_extraction(
        GstinExtractionSchema,
        (
            "Extract GSTIN values and reconciliation-ready company context from the input. "
            "Do not invent GSTINs or addresses. "
            "Use the company hint only as a hint, not as truth."
        ),
        f"Company hint: {company_hint}\n\nSupplier text:\n{text}",
    )
    return provider, result
