from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from dotenv import load_dotenv
import os
import agent_tools as at
import json
import re
from typing import Any, List, Optional

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  huggingface_hub not available. Install: pip install huggingface_hub")

DB_PATH = "salesdata.db"
load_dotenv()

# ============================================
# CONFIGURATION - Hugging Face API
# ============================================

USE_HUGGINGFACE_API = True  # Set to False to use only Groq

# Get API key from environment
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# Model Selection
# IMPORTANT: HF free tier (serverless) only supports specific models.
# These are confirmed to work with chat_completion on the free tier:

# Planner - good at instructions and JSON output
PLANNER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
# Other confirmed working options:
# - "Qwen/Qwen2.5-7B-Instruct"          (smaller, faster)
# - "microsoft/Phi-3.5-mini-instruct"   (small, fast)
# - "HuggingFaceH4/zephyr-7b-beta"      (older but reliable)
# - "google/gemma-2-9b-it"              (good quality)
# - "meta-llama/Llama-3.1-8B-Instruct" (needs HF access approval)

# SQL Model - for text-to-SQL generation
# NOTE: defog/sqlcoder-7b-2 is NOT available on free serverless tier
# Use a general model with SQL system prompt instead:
SQL_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
# Other confirmed working options:
# - "Qwen/Qwen2.5-72B-Instruct"  (best SQL quality on free tier)
# - "microsoft/Phi-3.5-mini-instruct"


# ============================================
# Hugging Face API Wrapper
# ============================================

class HuggingFaceAPIModel(BaseChatModel):
    """
    LangChain-compatible chat model wrapper for Hugging Face Inference API.
    Extends BaseChatModel so messages flow through _generate correctly.
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
        """Convert LangChain messages → HF chat format"""
        hf_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                hf_messages.append({"role": "assistant", "content": msg.content})
        return hf_messages
    
    def _generate(self, messages: List[BaseMessage], stop=None, **kwargs) -> ChatResult:
        """Core method called by LangChain - receives messages, returns ChatResult"""
        
        hf_messages = self._convert_messages(messages)
        
        # For SQL models, inject system prompt if not present
        if self.is_sql_model and not any(m["role"] == "system" for m in hf_messages):
            hf_messages.insert(0, {
                "role": "system",
                "content": "You are an expert SQL query generator for DuckDB. Return only valid SQL with no explanation."
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
                print(f"  [HINT]  '{self.model_name}' is not supported on the free serverless tier.")
                print(f"  [HINT]  Try one of these in agents.py:")
                print(f"          PLANNER_MODEL = 'Qwen/Qwen2.5-72B-Instruct'")
                print(f"          PLANNER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'")
                print(f"          PLANNER_MODEL = 'microsoft/Phi-3.5-mini-instruct'")
            content = f"ERROR: {e}"
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
    
    @property
    def _llm_type(self) -> str:
        return "huggingface-api"
    
    def bind_tools(self, tools: List[Any]) -> "HuggingFaceAPIModel":
        """Compatibility stub - HF models don't support native tool calling"""
        return self


# ============================================
# Model Loaders
# ============================================

_PLANNER_MODEL = None
_SQL_MODEL = None

def load_llm(model_type="executor"):
    """
    Load LLM based on type:
      "executor" → always Groq (supports bind_tools for structured tool calling)
      "planner"  → HF API if configured, else Groq
      "sql"      → HF API if configured, else Groq
    """
    global _PLANNER_MODEL, _SQL_MODEL
    
    # For tool calling, always use Groq (it supports bind_tools)
    if model_type == "executor":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0,
        )
    
    # For SQL generation, use HF API if available
    if model_type == "sql":
        if USE_HUGGINGFACE_API and HF_API_KEY and HF_HUB_AVAILABLE:
            if _SQL_MODEL is None:
                print(f"🗄️  Loading SQL Model via HF API: {SQL_MODEL}")
                _SQL_MODEL = HuggingFaceAPIModel(
                    model_name=SQL_MODEL,
                    api_key=HF_API_KEY,
                    temperature=0.1,
                    max_new_tokens=512,
                    is_sql_model=True,
                )
            return _SQL_MODEL
        else:
            # Fallback to Groq for SQL too
            if not HF_HUB_AVAILABLE and USE_HUGGINGFACE_API:
                print("⚠️  huggingface_hub not installed, using Groq")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0,
            )
    
    # For planning, use HF API if available, otherwise Groq
    if USE_HUGGINGFACE_API and HF_API_KEY and HF_HUB_AVAILABLE:
        if _PLANNER_MODEL is None:
            print(f"📋 Loading Planner Model via HF API: {PLANNER_MODEL}")
            _PLANNER_MODEL = HuggingFaceAPIModel(
                model_name=PLANNER_MODEL,
                api_key=HF_API_KEY,
                temperature=0.1,
                max_new_tokens=1024,
                is_sql_model=False,
            )
        return _PLANNER_MODEL
    else:
        # Fallback to Groq
        if not HF_API_KEY and USE_HUGGINGFACE_API:
            print("⚠️  HUGGINGFACE_API_KEY not found, using Groq")
        elif not HF_HUB_AVAILABLE and USE_HUGGINGFACE_API:
            print("⚠️  huggingface_hub not installed, using Groq")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0,
        )


# ============================================
# Response Handler
# ============================================


def handle_response(response):
    """
    Process LLM response and execute any tool calls.
    Returns: (is_done: bool, tool_messages: list)
      - is_done=True  → no tools called, response is final answer
      - is_done=False → tools were called, continue the loop
    """
    TOOL_MAP, _ = at.get_tools()
    tool_messages = []

    # ── Path 1: Structured tool calls (Groq native, preferred) ────────
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]
            tid  = call["id"]
            print(f"  [tool call]   {name}({args})")
            result = TOOL_MAP[name].invoke(args)
            print(f"  [tool result] {result[:200]}{'…' if len(result) > 200 else ''}")
            tool_messages.append(ToolMessage(content=result, tool_call_id=tid))
        return False, tool_messages

    # ── Path 2: Text-based tool calls (fallback) ───────────────────────
    # Matches patterns like: <function/run_query>{"sql": "..."}
    # or: run_query({"sql": "..."})
    content = getattr(response, "content", "") or ""
    if content:
        # Pattern 1: <function/tool_name>{...}</function>  (Groq text fallback format)
        pattern1 = r'<function/(\w+)>\s*(\{.*?\})\s*(?:</function>|(?=<function|$))'
        # Pattern 2: tool_name({"key": "value"})
        pattern2 = r'\b(run_query|get_schema|list_tables)\s*\(\s*(\{.*?\})\s*\)'

        calls_found = []
        for pat in (pattern1, pattern2):
            for m in re.finditer(pat, content, re.DOTALL | re.MULTILINE):
                tool_name = m.group(1)
                args_str  = m.group(2)
                try:
                    args = json.loads(args_str.strip())
                    calls_found.append((tool_name, args))
                except json.JSONDecodeError:
                    continue

        if calls_found:
            print(f"  [WARNING] Text-based tool calls detected — parsing fallback")
            for tool_name, args in calls_found:
                if tool_name in TOOL_MAP:
                    print(f"  [tool call]   {tool_name}({args})")
                    result = TOOL_MAP[tool_name].invoke(args)
                    print(f"  [tool result] {result[:200]}{'…' if len(result) > 200 else ''}")
                    # Use a fake tool_call_id so the message chain stays valid
                    fake_id = f"text_{tool_name}_{len(tool_messages)}"
                    tool_messages.append(ToolMessage(content=result, tool_call_id=fake_id))
            if tool_messages:
                return False, tool_messages

    # ── No tool calls → final answer ──────────────────────────────────
    return True, []


# ============================================
# Planning
# ============================================

PLANNER_PROMPT = at.get_planner_prompt()
_SCHEMA_CACHE = None

def _schemas() -> str:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = at.fetch_schemas()
    return _SCHEMA_CACHE


def get_plan(question: str, conversation_history: list = None) -> dict:
    """Generate query plan using planner model"""
    llm = load_llm("planner")
    
    history_str = ""
    if conversation_history:
        history_str = "\n".join(conversation_history)
    else:
        history_str = "(No previous conversation)"
    
    prompt = PLANNER_PROMPT.format(
        schemas=_schemas(),
        conversation_history=history_str,
        question=question
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Clean markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        if len(lines) > 2:
            raw = "\n".join(lines[1:-1])
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    try:
        plan = json.loads(raw)
        
        # Disable duplicate clarifications
        if plan.get("clarification_question") and "duplicate" in plan["clarification_question"].lower():
            plan["needs_clarification"] = False
            plan["notes"] = (plan.get("notes", "") + 
                           " Note: Handle duplicates by grouping.")
        
        # Safety validation
        if plan.get("needs_db", True) and not plan.get("needs_clarification"):
            has_empty_tables = not plan.get("tables") or len(plan.get("tables", [])) == 0
            
            if has_empty_tables:
                plan["needs_clarification"] = True
                if not plan.get("clarification_question"):
                    plan["clarification_question"] = (
                        "I need more information. Could you provide more details?"
                    )
        
        return plan
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Failed to parse plan: {e}")
        return {
            "needs_db": True,
            "tables": [],
            "joins": [],
            "conditions": [],
            "select_columns": [],
            "logic_summary": "",
            "needs_clarification": True,
            "clarification_question": "Could you rephrase that?",
            "notes": f"PARSE ERROR – {e}",
        }


# ============================================
# SQL Generation (Specialized Model via API)
# ============================================

def generate_sql_with_specialized_model(plan: dict, question: str) -> tuple:
    """
    Generate SQL using specialized text-to-SQL model via HF API.
    Returns: (success, result, sql)
    If HF model unavailable or errors, returns (False, None, None) → caller falls back to Groq.
    """
    if not USE_HUGGINGFACE_API or not HF_API_KEY or not HF_HUB_AVAILABLE:
        return False, None, None

    llm = load_llm("sql")
    schemas = _schemas()
    TOOL_MAP, _ = at.get_tools()

    system_msg = (
        "You are an expert SQL query generator for DuckDB. "
        "Return ONLY the raw SQL SELECT statement with no explanation, "
        "no markdown fences, no commentary."
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

        # Abort immediately if API returned an error
        if sql.startswith("ERROR:") or "not supported" in sql or "Bad request" in sql:
            print(f"  [SQL model] HF API unavailable — falling back to Groq")
            return False, None, None

        # Strip markdown fences if present
        if "```" in sql:
            sql = sql.split("```")[1]
            if sql.lower().startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()

        # Ensure it starts with SELECT
        if not sql.upper().lstrip().startswith("SELECT"):
            sql = "SELECT " + sql

        print(f"  [SQL] {sql[:100]}...")

        # Execute
        result = TOOL_MAP['run_query'].invoke({"sql": sql})
        if result.startswith("ERROR"):
            print(f"  [SQL error] {result[:120]}")
            # Feed error back into next attempt
            user_msg += f"\n\nPrevious attempt failed:\nSQL: {sql}\nError: {result}\nFix the SQL."
            continue

        print(f"  [SQL success]")
        return True, result, sql

    print("  [SQL model] All attempts failed — falling back to Groq")
    return False, None, None