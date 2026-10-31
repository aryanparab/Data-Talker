import os
import json
import httpx

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


async def complete(system, payload, json_mode=False):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Live questions are unavailable: the model key is not configured. Try a verified example."
        )
    messages = [{"role": "system", "content": system}]
    # Preserve conversational roles instead of burying prior questions inside a
    # catalog JSON blob. Historical SQL is context, never execution authority.
    current = json.loads(payload)
    for turn in current.pop("history", []):
        messages.append(
            {"role": "user", "content": json.dumps({"question": turn["question"]})}
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"sql": turn.get("sql", ""), "status": "previous query"}
                ),
            }
        )
    messages.append({"role": "user", "content": json.dumps(current)})
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": 3000 if json_mode else 900,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json=body,
        )
    if response.status_code != 200:
        raise RuntimeError(
            "The model provider is unavailable or rate limited. Your question has not changed the database. Try a verified example."
        )
    data = response.json()
    return data["choices"][0]["message"]["content"], data.get("usage", {})
