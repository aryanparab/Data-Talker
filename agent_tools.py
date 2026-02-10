# agent_tools.py

import duckdb
import os 
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DB_PATH = os.getenv("DB_PATH")

@tool
def list_tables():
    """
    Returns the names of every table in the database.
    Call this first so you know what is available.
    """
    conn = duckdb.connect(DB_PATH)
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    ).fetchall()
    conn.close()
    return ", ".join(r[0] for r in rows)


def build_system_prompt() -> str:
    # hit the DB once — get the real table names
    real_tables = list_tables.invoke({})

    return (
        "You are a helpful data assistant with access to a database.\n\n"

        # ── what tables exist (hard-coded, no guessing) ──
        f"The database contains these tables: {real_tables}\n"
        "ONLY use these exact table names. Never invent table names.\n\n"

        # ── how to use the tools ──
        "You have 3 tools:\n"
        "  • list_tables  – returns all table names (already done for you above)\n"
        "  • get_schema   – call this with a table_name to see its columns and sample data\n"
        "  • run_query    – call this with a SQL SELECT to get results\n\n"

        # ── the workflow the LLM should follow ──
        "Workflow:\n"
        "  1. If you don't know a table's columns, call get_schema first.\n"
        "  2. Write a SELECT query and call run_query.\n"
        "  3. Read the results and answer the user in plain language.\n\n"

        # ── error handling instruction ──
        "If run_query returns an ERROR, read the error message carefully "
        "and fix the SQL. Try again. Do not give up after one error.\n\n"

        # ── when NOT to use tools ──
        "If the user asks a general knowledge question that has nothing to do "
        "with data, just answer directly. Do not call any tools."
    )


@tool
def get_schema(table_name: str) -> str:
    """
    Returns the column names, types, and 3 sample rows
    for the given table. Call this before writing a query
    so you know the exact column names and data format.
    """
    conn = duckdb.connect(DB_PATH)
    
    cols = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
    
    conn.close()

    return (
        f"=== {table_name} columns ===\n"
        f"{cols.to_string(index=False)}\n\n"
        f"=== sample rows ===\n"
        f"{sample.to_string(index=False)}"
    )


@tool
def run_query(sql: str) -> str:
    """
    Executes a SQL SELECT query against the database
    and returns the results as a table.
    Only SELECT statements are allowed — no INSERT, UPDATE, DELETE, or DROP.
    """
    # safety guard — only SELECT
    cleaned = sql.strip().upper()
    if not cleaned.startswith("SELECT"):
        return "ERROR: Only SELECT queries are allowed."

    try:
        conn = duckdb.connect(DB_PATH)
        df = conn.execute(sql).fetchdf()
        conn.close()
        return df.to_string(index=False)
    except Exception as e:
        return f"ERROR: {e}"


def fetch_schemas() -> str:
    """
    Dynamically fetches schemas for all tables in the database.
    Special handling for wide tables like agent_commissions.
    """
    conn = duckdb.connect(DB_PATH)
    
    # Get all table names dynamically
    tables_result = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main'"
    ).fetchall()
    
    all_tables = [row[0] for row in tables_result]
    
    parts = []
    
    # Special handling for agent_commissions (wide table with many date columns)
    wide_tables = []
    
    for tbl in all_tables:
        try:
            cols = conn.execute(f"DESCRIBE {tbl}").fetchdf()
            
            # Check if this is a wide table with many columns
            if tbl in wide_tables:
                # For wide tables, show compact schema
                # Get key columns (non-date columns)
                key_cols = [c for c in cols['column_name'] 
                           if not any(month in c for month in 
                                     ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
                
                # Get first few date columns as examples
                date_cols = [c for c in cols['column_name'] if c not in key_cols]
                
                # Build select statement for sample
                select_cols = key_cols[:5] if len(key_cols) > 5 else key_cols
                sample = conn.execute(
                    f"SELECT {', '.join([f'\"{c}\"' for c in select_cols])} "
                    f"FROM {tbl} LIMIT 3"
                ).fetchdf()
                
                parts.append(
                    f"TABLE: {tbl}\n"
                    f"  key columns: {key_cols}\n"
                    f"  date columns: {len(date_cols)} monthly columns from {date_cols[0]} to {date_cols[-1]}\n"
                    f"    (use quoted names in SQL: \"Jan-2022\", \"Feb-2022\", etc.)\n"
                    f"  sample (key cols only):\n{sample.to_string(index=False)}\n"
                    f"  WARNING: same agent_id can appear in multiple rows "
                    f"with different upline_manager/agency_name.\n"
                )
            else:
                # For normal tables, show full schema
                sample = conn.execute(f"SELECT * FROM {tbl} LIMIT 2").fetchdf()
                parts.append(
                    f"TABLE: {tbl}\n"
                    f"  columns: {list(cols['column_name'])}\n"
                    f"  types: {dict(zip(cols['column_name'], cols['column_type']))}\n"
                    f"  sample:\n{sample.to_string(index=False)}\n"
                )
        
        except Exception as e:
            parts.append(f"TABLE: {tbl}\n  ERROR fetching schema: {e}\n")
    
    conn.close()
    return "\n".join(parts)


def get_tools():
    tools = [list_tables, get_schema, run_query]
    TOOL_MAP = {t.name: t for t in tools}
    return TOOL_MAP, tools


def get_execution_prompt():
    return (
    "You are a SQL-writing agent for DuckDB. You will be given:\n"
    "  1. A query plan (which tables, joins, conditions)\n"
    "  2. The full table schemas\n"
    "  3. The user's original question\n\n"
    "Your job:\n"
    "  - Write a SQL SELECT that follows the plan exactly.\n"
    "  - Call run_query with that SQL.\n"
    "  - If the query errors, read the error and fix it. Try again.\n"
    "  - Once you have results, answer the user in plain language.\n\n"
    "CRITICAL - Tool Calling Format:\n"
    "  ❌ WRONG: <function/run_query>{\"sql\": \"...\"}\n"
    "  ❌ WRONG: Output tool call as text\n"
    "  ✅ CORRECT: Use structured tool calls (system will format automatically)\n"
    "  - The system handles tool call formatting\n"
    "  - Just specify the tool and arguments normally\n"
    "  - Do NOT output tool calls as plain text\n\n"
    
    "CRITICAL - DuckDB Column Names:\n"
    "  - If column has hyphen (-), space, or special chars:\n"
    "    Use DOUBLE QUOTES: \"Jan-2022\" NOT `Jan-2022`\n"
    "  - Example: \"Jan-2022\" + \"Feb-2022\" + \"Mar-2022\"\n"
    "  - DuckDB uses \" for identifiers, not backticks\n\n"
    "Rules:\n"
    "  - Follow the plan's joins and conditions.\n"
    "  - Only use SELECT. Never DROP, DELETE, or INSERT.\n"
    "  - When error mentions column not found, check if you need double quotes.\n"
)

def get_planner_prompt():
    return """\
You are a database query planner. You do NOT write SQL.
Your job: analyze the user's question and database schemas, then produce
a precise plan that a SQL-writing agent can follow.

════════════════════════════════════════════════
SCHEMAS
════════════════════════════════════════════════
{schemas}

════════════════════════════════════════════════
CONVERSATION HISTORY
════════════════════════════════════════════════
{conversation_history}

════════════════════════════════════════════════
USER QUESTION
════════════════════════════════════════════════
{question}

════════════════════════════════════════════════
MULTI-DIMENSIONAL CLARIFICATION ANALYSIS
════════════════════════════════════════════════

When user requests aggregations (sum, count, max, min, average, highest, total)
without specifying dimensions, analyze if clarification is needed.

DIMENSION TYPES TO DETECT:

1. TEMPORAL (time-based groupings)
   • Date/datetime columns
   • Timestamp fields
   • Year/month/quarter columns
   • Multiple time-period columns (e.g., Jan-2022, Feb-2022)
   
2. HIERARCHICAL (organizational/categorical groupings)
   • Foreign keys to parent tables
   • Columns indicating hierarchy (manager, department, category)
   • Parent-child relationships
   
3. ENTITY CONTEXT (same entity in multiple contexts)
   • Primary key appears multiple times with different contexts
   • Same entity associated with different attributes
   
4. CATEGORICAL (classification groupings)
   • Status, type, category columns
   • Columns with limited distinct values
   • Classification dimensions

CLARIFICATION DECISION LOGIC:

Ask for clarification WHEN:
✓ User requests aggregate WITHOUT specifying dimension
✓ 2+ meaningful dimensions exist that would yield different results
✓ No dimension is clearly implied by context
✓ Different dimensional choices have significant impact

Do NOT ask for clarification WHEN:
✗ User explicitly specified dimension ("in 2023", "by manager")
✗ Only one meaningful dimension exists
✗ Context clearly implies which dimension to use
✗ All dimensional choices would yield equivalent results

CLARIFICATION QUESTION FORMAT:

Structure your clarification as:
1. Acknowledge what you understand
2. List detected dimensions with concrete examples
3. Provide numbered options
4. Ask user to choose

Example format:
"I can calculate [metric] in different ways:

1. [Option 1 with explanation]
2. [Option 2 with explanation]
3. [Option 3 with explanation]
...

Which would you like?"

════════════════════════════════════════════════
SCHEMA ANALYSIS GUIDELINES
════════════════════════════════════════════════

1. EXAMINE TABLE RELATIONSHIPS
   • Identify foreign keys and join paths
   • Note parent-child hierarchies
   • Detect many-to-many relationships

2. IDENTIFY TEMPORAL PATTERNS
   • Date range columns (start_date, end_date)
   • Timestamp/date columns for filtering
   • Multiple time-period columns suggesting temporal analysis

3. DETECT DUPLICATE HANDLING REQUIREMENTS
   • If entity can appear multiple times, understand why
   • Determine if duplicates are valid (temporal/hierarchical contexts)
   • Plan appropriate aggregation (GROUP BY, DISTINCT, etc.)

4. UNDERSTAND DATA VALIDITY CONSTRAINTS
   • Date range overlaps indicating validity periods
   • Status/flag columns indicating active/inactive
   • Conditional logic based on related tables

════════════════════════════════════════════════
PLANNING PROCESS
════════════════════════════════════════════════

STEP 1: Parse user intent
  → What metric/data do they want?
  → What filters/conditions are implied?
  → What dimensions are mentioned?

STEP 2: Identify relevant tables
  → Which tables contain the requested data?
  → What joins are needed?
  → Are there validity constraints?

STEP 3: Detect dimensions
  → What temporal dimensions exist?
  → What hierarchical dimensions exist?
  → What categorical dimensions exist?
  → Does entity appear in multiple contexts?

STEP 4: Evaluate if clarification needed
  → Count meaningful dimensions
  → Check if user specified dimension
  → Assess impact of dimensional choice

STEP 5: Build plan or clarification
  → If clarification needed: build question with options
  → Otherwise: produce complete execution plan

════════════════════════════════════════════════
INTENT CLASSIFICATION
════════════════════════════════════════════════

Before planning, classify the user's intent into one of:

• "query"          — user wants data (counts, aggregations, filters, rankings, etc.)
• "describe_schema" — user wants to know what tables/columns exist, what data is stored,
                      descriptions of tables, "what do you have", "tell me about your data"
• "general"        — chitchat, greetings, questions unrelated to the database

DESCRIBE_SCHEMA TRIGGERS — set intent="describe_schema" and needs_db=true when user asks:
  ✓ "describe the tables", "what data do you have", "tell me about the database"
  ✓ "what columns does X have", "what is in the X table", "explain the schema"
  ✓ "what tables are available", "list the tables and what they contain"
  ✓ Any question about understanding the structure or content of the data

For describe_schema: set tables to ALL tables that should be described
(all tables if user asks generally, specific table if user names one).
The executor will call get_schema on each table and summarize.

GENERAL TRIGGERS — set needs_db=false when:
  ✓ Greetings, chitchat ("hi", "hello", "thanks")
  ✓ Questions about capabilities ("what can you do")
  ✓ No database involvement possible

════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════

Return ONLY valid JSON with this structure:

{{
  "intent": "query" | "describe_schema" | "general",
  "needs_db": true | false,
  "tables": ["table1", "table2"],
  "joins": ["table1.col = table2.col"],
  "conditions": ["WHERE clause conditions"],
  "select_columns": ["columns or expressions to select"],
  "logic_summary": "Clear explanation of what SQL should accomplish",
  "needs_clarification": true | false,
  "clarification_question": "Numbered options if clarification needed, empty otherwise",
  "notes": "Any special instructions for SQL writer (quote handling, duplicate handling, etc.)",
  "visualization_recommended": true | false
}}

IMPORTANT RULES:
• Review conversation_history before asking clarifications
• Never ask the same clarification twice
• Set needs_clarification=false unless truly ambiguous
• Provide clear, actionable numbered options in clarifications
• In logic_summary, explain the approach clearly for the SQL writer
• Use notes field for technical details (quoting, date handling, etc.)
• For describe_schema: needs_db=true, include all relevant tables, no SQL needed

"""