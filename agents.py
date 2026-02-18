from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from dotenv import load_dotenv
import os
import time
import agent_tools as at
import json
import re
from typing import Any, List

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  huggingface_hub not available. Install: pip install huggingface_hub")

load_dotenv()

DB_PATH = "salesdata.db"

# ============================================
# CONFIGURATION
# ============================================

# Set to False to route all models through Groq only
USE_HUGGINGFACE_API = True

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
groq_key   = os.getenv("GROQ_API_KEY")

# Planner model — needs strong instruction-following and JSON output
# Confirmed working on HF free serverless tier:
#   "Qwen/Qwen2.5-72B-Instruct"         (best quality)
#   "Qwen/Qwen2.5-7B-Instruct"          (smaller, faster)
#   "microsoft/Phi-3.5-mini-instruct"   (lightweight)
#   "google/gemma-2-9b-it"
PLANNER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# SQL model — text-to-SQL generation
# Note: defog/sqlcoder-7b-2 is NOT available on the free serverless tier
# Confirmed working alternatives:
#   "Qwen/Qwen2.5-Coder-7B-Instruct"   (good SQL quality, fast)
#   "Qwen/Qwen2.5-72B-Instruct"        (best SQL quality on free tier)
SQL_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


# ============================================
# HUGGING FACE API WRAPPER
# ============================================

class HuggingFaceAPIModel(BaseChatModel):
    """
    LangChain-compatible wrapper for the Hugging Face Inference API.
    Extends BaseChatModel so it integrates cleanly with LangChain message flows.
    Used for planner and SQL generation roles where native tool calling is not needed.
    """

    model_name: str
    api_key: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    is_sql_model: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _get_client(self):
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub is required. Run: pip install huggingface_hub")
        return InferenceClient(token=self.api_key)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain message objects to HF chat completion format."""
        hf_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_messages.append({"role": "system",    "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                hf_messages.append({"role": "assistant", "content": msg.content})
        return hf_messages

    def _generate(self, messages: List[BaseMessage], stop=None, **kwargs) -> ChatResult:
        """Core LangChain method — converts messages, calls HF API, returns ChatResult."""
        hf_messages = self._convert_messages(messages)

        # SQL models get a default system prompt injected if none was provided
        if self.is_sql_model and not any(m["role"] == "system" for m in hf_messages):
            hf_messages.insert(0, {
                "role": "system",
                "content": "You are an expert SQL query generator for DuckDB. Return only valid SQL with no explanation.",
            })

        try:
            client = self._get_client()
            response = client.chat_completion(
                messages=hf_messages,
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content

        except Exception as e:
            err = str(e)
            print(f"  [ERROR] HuggingFace API error: {e}")
            if "not a chat model" in err or "model_not_supported" in err:
                print(f"  [HINT] '{self.model_name}' is unsupported on the free serverless tier.")
                print(f"  [HINT] Try: PLANNER_MODEL = 'Qwen/Qwen2.5-72B-Instruct'")
            content = f"ERROR: {e}"

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "huggingface-api"

    def bind_tools(self, tools: List[Any]) -> "HuggingFaceAPIModel":
        """Stub — HF serverless models do not support native tool calling."""
        return self


# ============================================
# MODEL LOADERS
# ============================================

# Module-level singletons — models are instantiated once and reused across calls
_PLANNER_MODEL = None
_SQL_MODEL     = None


def load_llm(model_type: str = "executor"):
    """
    Return the appropriate LLM for the given role:
      "executor" → always Groq (required for native bind_tools / structured tool calls)
      "planner"  → HF API if configured, otherwise falls back to Groq
      "sql"      → HF API if configured, otherwise falls back to Groq
    """
    global _PLANNER_MODEL, _SQL_MODEL

    # Executor always uses Groq — it is the only role that needs bind_tools
    if model_type == "executor":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0,
        )

    # SQL generation — prefer HF specialist model, fall back to Groq
    if model_type == "sql":
        if USE_HUGGINGFACE_API and HF_API_KEY and HF_HUB_AVAILABLE:
            if _SQL_MODEL is None:
                print(f"🗄️  Loading SQL model via HF API: {SQL_MODEL}")
                _SQL_MODEL = HuggingFaceAPIModel(
                    model_name=SQL_MODEL,
                    api_key=HF_API_KEY,
                    temperature=0.1,
                    max_new_tokens=512,
                    is_sql_model=True,
                )
            return _SQL_MODEL
        if not HF_HUB_AVAILABLE and USE_HUGGINGFACE_API:
            print("⚠️  huggingface_hub not installed, using Groq for SQL")
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0)

    # Planner — prefer HF model for better JSON output quality, fall back to Groq
    if USE_HUGGINGFACE_API and HF_API_KEY and HF_HUB_AVAILABLE:
        if _PLANNER_MODEL is None:
            print(f"📋 Loading planner model via HF API: {PLANNER_MODEL}")
            _PLANNER_MODEL = HuggingFaceAPIModel(
                model_name=PLANNER_MODEL,
                api_key=HF_API_KEY,
                temperature=0.1,
                max_new_tokens=1024,
                is_sql_model=False,
            )
        return _PLANNER_MODEL

    if not HF_API_KEY and USE_HUGGINGFACE_API:
        print("⚠️  HUGGINGFACE_API_KEY not set, using Groq for planner")
    elif not HF_HUB_AVAILABLE and USE_HUGGINGFACE_API:
        print("⚠️  huggingface_hub not installed, using Groq for planner")
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0)


# ============================================
# TOOL CALL HANDLER
# ============================================

def handle_response(response):
    """
    Process an LLM response and execute any tool calls found in it.

    Returns:
        (is_done: bool, tool_messages: list)
        - is_done=True  → no tools were called; response is the final answer
        - is_done=False → tools were called; caller should extend messages and loop
    """
    TOOL_MAP, _ = at.get_tools()
    tool_messages = []

    # Path 1: Structured tool calls (Groq native — preferred path)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            name   = call["name"]
            args   = call["args"]
            tid    = call["id"]
            print(f"  [tool call]   {name}({args})")
            result = TOOL_MAP[name].invoke(args)
            print(f"  [tool result] {result[:200]}{'…' if len(result) > 200 else ''}")
            tool_messages.append(ToolMessage(content=result, tool_call_id=tid))
        return False, tool_messages

    # Path 2: Text-based tool calls (fallback for models that emit calls as plain text)
    # Handles two formats:
    #   <function/run_query>{"sql": "..."}
    #   run_query({"sql": "..."})
    content = getattr(response, "content", "") or ""
    if content:
        pattern1 = r'<function/(\w+)>\s*(\{.*?\})\s*(?:</function>|(?=<function|$))'
        pattern2 = r'\b(run_query|get_schema|list_tables)\s*\(\s*(\{.*?\})\s*\)'

        calls_found = []
        for pat in (pattern1, pattern2):
            for m in re.finditer(pat, content, re.DOTALL | re.MULTILINE):
                try:
                    args = json.loads(m.group(2).strip())
                    calls_found.append((m.group(1), args))
                except json.JSONDecodeError:
                    continue

        if calls_found:
            print("  [WARNING] Text-based tool calls detected — using parse fallback")
            for tool_name, args in calls_found:
                if tool_name in TOOL_MAP:
                    print(f"  [tool call]   {tool_name}({args})")
                    result = TOOL_MAP[tool_name].invoke(args)
                    print(f"  [tool result] {result[:200]}{'…' if len(result) > 200 else ''}")
                    # Fake ID is needed to keep the LangChain message chain valid
                    fake_id = f"text_{tool_name}_{len(tool_messages)}"
                    tool_messages.append(ToolMessage(content=result, tool_call_id=fake_id))
            if tool_messages:
                return False, tool_messages

    # No tool calls found — treat the response as the final answer
    return True, []


# ============================================
# QUERY PLANNER
# ============================================

PLANNER_PROMPT = at.get_react_planner_prompt()
_SCHEMA_CACHE  = None


def _schemas() -> str:
    """Return the full DB schema string, cached after first load."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = at.fetch_schemas()
    return _SCHEMA_CACHE


def get_plan(question: str, conversation_history: list = None, context_check: str = "") -> dict:
    """
    Send the question and conversation context to the planner model and return
    a structured JSON plan that the executor will use to generate SQL.

    context_check: real DB lookup result injected before planning so the planner
    never needs to guess whether an agent exists in multiple contexts.

    Retries up to 3 times on empty responses (rate limits or transient API errors).
    Returns a safe error plan if all retries fail, so the user sees a real message
    rather than a misleading clarification prompt.
    """
    llm = load_llm("planner")
    history_str = "\n".join(conversation_history) if conversation_history else "(No previous conversation)"

    prompt = PLANNER_PROMPT.format(
        schemas=_schemas(),
        conversation_history=history_str,
        question=question,
        instructions=at.get_instruction_set(),
        context_check=context_check if context_check else "(No entity context lookup performed)",
    )

    # Retry on empty response — Groq/HF can return blank on rate limit or timeout
    raw = ""
    for attempt in range(3):
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw:
            break
        wait = 2 ** attempt  # 1s, 2s, 4s backoff
        print(f"  [PLANNER] Empty response (attempt {attempt + 1}/3), retrying in {wait}s...")
        time.sleep(wait)

    # Strip markdown fences if the model wrapped its JSON output
    if raw.startswith("```"):
        lines = raw.split("\n")
        if len(lines) > 2:
            raw = "\n".join(lines[1:-1])
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    try:
        plan = json.loads(raw)

        # Suppress duplicate-handling clarifications — use GROUP BY in SQL instead
        if plan.get("clarification_question") and "duplicate" in plan["clarification_question"].lower():
            plan["needs_clarification"] = False
            plan["notes"] = plan.get("notes", "") + " Note: Handle duplicates by grouping."

        # Safety check: if DB is needed but no tables were identified, ask for clarification
        if plan.get("needs_db", True) and not plan.get("needs_clarification"):
            if not plan.get("tables"):
                plan["needs_clarification"] = True
                plan.setdefault("clarification_question", "I need more information. Could you provide more details?")

        return plan

    except json.JSONDecodeError as e:
        # Return a safe fallback that surfaces a real error rather than
        # routing into the clarification loop with a confusing message
        print(f"  [ERROR] Failed to parse plan: {e}\n  Raw: '{raw[:200]}'")
        return {
            "needs_db": False,
            "needs_clarification": False,
            "intent": "general",
            "tables": [],
            "error_message": "I'm having trouble processing that right now. Please try again in a moment.",
        }


# ============================================
# SQL GENERATION (HF SPECIALIST MODEL)
# ============================================

def generate_sql_with_specialized_model(plan: dict, question: str) -> tuple:
    """
    Attempt SQL generation using the HF specialist model.
    Retries up to 3 times, feeding errors back to the model for self-correction.

    Returns:
        (True,  result_str, sql_str)  on success
        (False, None,       None)     if unavailable or all attempts fail — caller falls back to Groq
    """
    if not USE_HUGGINGFACE_API or not HF_API_KEY or not HF_HUB_AVAILABLE:
        return False, None, None

    llm         = load_llm("sql")
    schemas     = _schemas()
    TOOL_MAP, _ = at.get_tools()

    system_msg = (
        "You are an expert SQL query generator for DuckDB. "
        "Return ONLY the raw SQL SELECT statement — no explanation, no markdown, no commentary."
    )

    user_msg = (
        f"Database schema:\n{schemas}\n\n"
        f"Tables to use: {', '.join(plan.get('tables', []))}\n"
        f"Joins: {', '.join(plan.get('joins', [])) or 'none'}\n"
        f"Conditions: {', '.join(plan.get('conditions', [])) or 'none'}\n"
        f"Logic: {plan.get('logic_summary', '')}\n"
        f"Notes: {plan.get('notes', '')}\n\n"
        f"Question: {question}\n\n"
        "Write only the SELECT statement."
    )

    for attempt in range(3):
        print(f"  [SQL model] Attempt {attempt + 1}/3 via HF API...")

        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])
        sql = response.content.strip()

        # Hard fail — HF API itself is down or the model is not supported
        if sql.startswith("ERROR:") or "not supported" in sql or "Bad request" in sql:
            print("  [SQL model] HF API unavailable — falling back to Groq")
            return False, None, None

        # Strip markdown fences if the model included them
        if "```" in sql:
            sql = sql.split("```")[1]
            if sql.lower().startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()

        # Ensure query starts with SELECT (model occasionally omits it)
        if not sql.upper().lstrip().startswith("SELECT"):
            sql = "SELECT " + sql

        print(f"  [SQL] {sql[:100]}...")

        result = TOOL_MAP["run_query"].invoke({"sql": sql})
        if result.startswith("ERROR"):
            print(f"  [SQL error] {result[:120]}")
            # Feed the error back so the model can self-correct on the next attempt
            user_msg += f"\n\nPrevious attempt failed:\nSQL: {sql}\nError: {result}\nFix the SQL."
            continue

        print("  [SQL success]")
        return True, result, sql

    print("  [SQL model] All 3 attempts failed — falling back to Groq")
    return False, None, None