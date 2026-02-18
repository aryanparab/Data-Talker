# agent_tools.py

import duckdb
import os 
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DB_PATH = os.getenv("DB_PATH") or os.path.join(os.path.dirname(__file__), "salesdata.db")

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
    if not cleaned.startswith("SELECT") and not cleaned.startswith("WITH") and not cleaned.startswith("COALESCE"):
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


def get_instruction_set():
    return """

INVALID SALES - Finding Orders Outside Training Windows:

Concept: An order is 'invalid' if it doesn't match ANY of the salesperson's training windows.

Logic:
- A salesperson can have multiple training windows (multiple rows in training table)
- An order is valid if it falls within ANY training window
- An order is invalid if it falls within NO training windows

How to identify:
- Use LEFT JOIN from orders to training
- Include BOTH the foreign key AND date range in the JOIN condition
- Filter WHERE training.id IS NULL to get orders with no match
- This avoids counting the same order multiple times

Tables: orders, salesperson, training
Key relationships: 
  orders.salesperson_id → salesperson.id
  orders.salesperson_id → training.salesperson_id (with date overlap check)

  UNDERSTANDING VALID vs TOTAL SALES:

Only filter by training windows when explicitly asked about:
- "valid sales"
- "invalid sales" 
- "bonus calculation" (bonuses are based on valid sales)

────────────────────────────────────────────────────────────────

BONUS CALCULATION:

Concept: Bonuses are awarded based on total valid sales performance, not employee attributes.

Important: bonus_pay is a standalone lookup table with NO foreign keys to other tables.
  - bonus_pay.tier = dollar threshold (e.g., $1500 means you need $1500 in sales)
  - bonus_pay.bonus = dollar amount awarded
  - bonus_pay.id is just a row identifier, not a foreign key

Logic:
1. Calculate total VALID sales (orders within training windows) for the person/year
2. Compare that total to bonus_pay.tier thresholds for that year
3. Award the HIGHEST bonus where sales_total >= tier

How to calculate:
- First, sum valid sales amounts (use LEFT JOIN pattern from above, filter IS NOT NULL)
- Then, match that sum to bonus_pay tiers where sum >= tier
- Use MAX(bonus) to get the highest tier qualified for

Tables: orders, salesperson, training, bonus_pay
Key relationships:
  orders.salesperson_id → salesperson.id
  orders.salesperson_id → training.salesperson_id (with date overlap)
  bonus_pay has NO foreign keys - it's compared to calculated sales totals

Avoid:
- Don't join bonus_pay.id to other tables (it's not a foreign key)
- Don't join bonus_pay directly to orders/training (creates cartesian products)
- Calculate sales total first, then compare to bonus tiers

UNDERSTANDING DATA RELATIONSHIPS FROM COLUMN NAMES:

Foreign Key Pattern:
- Column ending in `_id` typically references another table's `id` column
- Example: salesperson_id references salesperson.id

Hierarchical/Contextual Columns:
- When a table has multiple descriptive columns (like agent_name, upline_manager, agency_name),
  each row represents that entity in a specific context
- Same entity with different context columns = multiple relationships

Clarification Decision:
When querying a specific entity that exists in multiple contexts:

CRITICAL: Before planning, CHECK if the entity appears multiple times:
- If querying by name (agent_name, salesperson name, etc.), check if it appears in multiple rows
- For agent_commissions: Check if same agent_name has different upline_manager or agency_name
- Conceptually: Would SELECT COUNT(DISTINCT upline_manager) WHERE agent_name = 'X' return > 1?
- If yes, there are multiple contexts that need clarification

ASK for clarification if ALL of these are true:
- Entity exists in multiple contexts (different upline_manager or agency_name)
- User query is simple/direct: "How much commission did X make?", "What are X's earnings?"
- No aggregation keywords: "total", "all", "combined", "sum"
- No context specified: "under [manager]", "at [agency]"
- No breakdown requested: "by", "per", "breakdown"

DON'T ASK if ANY of these are true:
- Aggregation implied: "total", "all", "sum" → sum across all contexts
- Context specified: "Alex Smith under [manager name]" → use that context
- Breakdown requested: "by manager", "per agency" → show all with grouping
- Comparison requested: "compare" → show all contexts
- Only one context exists for this entity

When clarifying, provide numbered options:
"I found Alex Smith under multiple managers. Which would you like?
1. Under [actual manager 1] at [actual agency 1]
2. Under [actual manager 2] at [actual agency 2]
3. Total across all managers"

"""

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
CONVERSATION HISTORY (Use this for context!)
════════════════════════════════════════════════
{conversation_history}

**IMPORTANT - Using Conversation Context:**

If the current question references previous messages:
- "now for Alice" → Look at previous question, apply same logic to Alice
- "what about 2023?" → Look at previous question, change the year to 2023
- "and invalid sales?" → Look at previous question, show invalid sales instead
- "show me the breakdown" → Look at previous answer, provide more detail

**Context Resolution Examples:**

Previous: "User: What are sales for Bob?"
Current: "User: Now for Alice"
→ Interpret as: "What are sales for Alice?"

Previous: "User: Show valid sales for Bob in 2024"
Previous: "Assistant: Bob had $1,500 in valid sales"
Current: "User: What about invalid sales?"
→ Interpret as: "Show invalid sales for Bob in 2024"

Previous: "User: How much did Bob make?"
Previous: "Assistant: Bob made $5,000 in sales"
Current: "User: What about 2023?"
→ Interpret as: "How much did Bob make in 2023?"

════════════════════════════════════════════════
USER QUESTION
════════════════════════════════════════════════
{question}

"CRITICAL - User Instruction set:\n"
{instructions}
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
  → **CRITICAL: If query mentions specific entity by name (e.g., "Alex Smith"), check if it appears multiple times**
  → For agent_commissions: Does agent have multiple upline_manager or agency_name values?
  → If duplicate contexts exist AND user didn't specify which → needs_clarification = true
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


# CONDENSED REACT-STYLE PLANNER PROMPT

def get_react_planner_prompt():
   
    """
    ReAct planner reorganized to match flowchart logic.
    Clarification and business logic placed where actually used.
    """
    return """\
You are a query planner using ReAct (Reasoning + Acting).

Your job: Think step-by-step, then output a JSON plan.

════════════════════════════════════════════════════════════════════
DATABASE SCHEMAS
════════════════════════════════════════════════════════════════════
{schemas}

════════════════════════════════════════════════════════════════════
CONVERSATION CONTEXT (Last 5 turns)
════════════════════════════════════════════════════════════════════
{conversation_history}

════════════════════════════════════════════════════════════════════
USER QUESTION
════════════════════════════════════════════════════════════════════
{question}

════════════════════════════════════════════════════════════════════
CRITICAL BUSINESS RULES
════════════════════════════════════════════════════════════════════
{instructions}

════════════════════════════════════════════════════════════════════
REACT PLANNING PROCESS - FOLLOW THIS EXACT ORDER!
════════════════════════════════════════════════════════════════════

**STEP 1: CHECK FOR PREVIOUS CLARIFICATION**

Look at conversation_history for my last message.

Does it contain:
- "Which [X]?" with numbered options (1. 2. 3.)
- "Select", "Choose", or clarification language

If YES → User is likely answering my clarification!

**Clarification Resolution Logic:**
```
My previous: "Which Alex Smith?
              1. Under Manager A at Summit
              2. Under Manager B at Peak
              3. Total"

User says:        I resolve to:
"option 1"    →  Manager A at Summit
"1"           →  Manager A at Summit  
"the first"   →  Manager A at Summit
"manager A"   →  Manager A (extract from text)
"summit"      →  Summit agency (extract from text)
"total"       →  No filtering (aggregate all)

User says:        I do:
"yes"         →  Unclear! Ask again with clearer options
"no"          →  Unclear! Ask again
```

If resolved → Build plan with that context, set needs_clarification = false
If unclear → Ask clarification again with numbered choices

**STEP 2: ANALYZE INTENT**

What is the user asking for?
- **Data retrieval**: "show me sales", "what are commissions"
- **Visualization request**: "create a chart", "histogram", "plot", "graph"
- **Schema info**: "what tables exist", "show me columns"
- **General chat**: "hello", "thanks"

**CRITICAL: Detect Explicit Visualization Requests**

If user uses ANY of these words, set explicit_visualization = true:
- "chart", "graph", "plot", "visualize", "visualization"
- "histogram", "bar chart", "line graph", "pie chart", "scatter plot"
- "heatmap", "diagram", "show me a chart", "create a graph"
- "draw a", "generate a chart", "make a plot"

This ensures auto-generation of visualizations when explicitly requested.

**STEP 3: DETECT DOMAIN**

Based on keywords, which domain?

**Domain 1: Salesperson Performance**
- Keywords: "sales", "orders", "training", "bonus", "sold"
- Tables: orders, salesperson, training, bonus_pay
- Purpose: Track product sales and salesperson bonuses

**Domain 2: Agent Commissions**
- Keywords: "commission", "agent", "manager", "agency", "upline"
- Tables: agent_commissions
- Purpose: Track insurance agent monthly commissions

**CRITICAL: NEVER mix these domains!**

**Domain Ambiguity Example:**
```
User: "Show me Bob's performance"

Think: "performance" is vague.
Check: Is "Bob" in salesperson table? (assume yes)
Check: Is "Bob" in agent_commissions table? (assume no)
Result: Use salesperson domain (only one match)
```

**STEP 4: CONTEXT RESOLUTION (Follow-ups)**

Is this a follow-up to a previous question?

Indicators:
- "now for Alice" (same query, different person)
- "what about 2023?" (same query, different year)
- "and invalid sales?" (same query, different metric)

If follow-up:
- Extract context from previous messages (entity, year, metric)
- Apply to current question
- Maintain any resolved clarification context

**STEP 5: TABLE SELECTION**

Which tables do I need for this query?

**For each table, ask:**
- Does this table have the data requested?
- What columns will I use?
- How do I join to other tables?

**CRITICAL DECISION: Training Table**

⚠️ **ONLY use training table when user EXPLICITLY asks for:**
1. "valid sales" / "invalid sales"
2. "bonus" (bonuses are based on valid sales)

**DO NOT use training table for:**
- General "sales" queries
- "total sales"
- "sales for [person]"

**Examples:**
```
Query: "Sales for Bob in 2024"
Tables: orders, salesperson
Join: orders.salesperson_id = salesperson.ID
Filter: Name = 'Bob', YEAR(order_date) = 2024
Training: NO ❌

Query: "Valid sales for Bob in 2024"
Tables: orders, salesperson, training
Join: orders.salesperson_id = salesperson.ID
      AND orders.salesperson_id = training.salesperson_id
      AND order_date BETWEEN Start_date AND End_date
Filter: Name = 'Bob', YEAR(order_date) = 2024
Training: YES ✅

Query: "Bob's bonus for 2024"
Tables: orders, salesperson, training, bonus_pay
Why training?: Bonuses use VALID sales only
Training: YES ✅
```

**Bonus Calculation Logic (when user asks for "bonus"):**

Bonuses are awards based on valid sales performance.

**Process:**
1. Calculate total VALID sales (orders within training windows)
2. Compare that total to bonus_pay.tier thresholds
3. Award the HIGHEST bonus where total >= tier

**Important about bonus_pay table:**
- It's a LOOKUP table (NOT joined by ID!)
- bonus_pay.tier = dollar threshold (e.g., $1500 in sales needed)
- bonus_pay.bonus = dollar amount awarded
- bonus_pay.id is just a row identifier (NOT a foreign key)

**How to calculate (SINGLE QUERY with CTE):**
```sql
-- Bonus calculation is ONE query with two parts:
WITH valid_sales AS (
  -- Part 1: Calculate total valid sales
  SELECT SUM(o.Amount) as total_valid
  FROM orders o
  JOIN salesperson s ON o.salesperson_id = s.ID
  JOIN training t ON o.salesperson_id = t.salesperson_id
    AND o.order_date BETWEEN t.Start_date AND t.End_date
  WHERE s.Name = 'Bob' AND YEAR(o.order_date) = 2024
)
-- Part 2: Find highest qualifying bonus (uses Part 1 result)
SELECT MAX(b.bonus) as earned_bonus
FROM bonus_pay b, valid_sales vs
WHERE b.tier <= vs.total_valid
  AND b.Year = 2024;
```

**CRITICAL:** This is ONE query, not two! The CTE (WITH clause) calculates sales, then the main SELECT finds the bonus.

**Alternative (using subquery):**
```sql
SELECT MAX(b.bonus) as earned_bonus
FROM bonus_pay b
WHERE b.tier <= (
  SELECT SUM(o.Amount)
  FROM orders o
  JOIN salesperson s ON o.salesperson_id = s.ID
  JOIN training t ON o.salesperson_id = t.salesperson_id
    AND o.order_date BETWEEN t.Start_date AND t.End_date
  WHERE s.Name = 'Bob' AND YEAR(o.order_date) = 2024
)
AND b.Year = 2024;
```


**Common mistakes to avoid:**
- ❌ Don't join bonus_pay.id to other tables
- ❌ Don't join bonus_pay directly to orders (creates cartesian product!)
- ❌ Don't write TWO separate queries (use CTE or subquery instead)
- ✅ Calculate sales total and match to tiers in ONE query using WITH or subquery

**STEP 6: CLARIFICATION CHECK**

Now that I know which tables I need, do I need clarification?

**Sub-step 6a: Check if entity exists in multiple contexts**

**CRITICAL: Do NOT invent or assume context data.**
Real context data has already been looked up and injected below (if applicable):

════════════════════════════════════════════════════════════════════
CONTEXT CHECK RESULTS (Real data from DB — use this, do not guess)
════════════════════════════════════════════════════════════════════
{context_check}

If context_check is empty → no agent found in multiple contexts → skip clarification
If context_check has multiple rows → ask clarification using the ACTUAL names shown above
════════════════════════════════════════════════════════════════════

**Sub-step 6b: Decide if clarification is needed**

Ask clarification ONLY if ALL of these are true:
✓ Multiple contexts exist (you checked and found >1 row)
✓ User didn't specify which context ("under manager A", "at Summit")
✓ User didn't say "total" or "all" or "combined"
✓ User didn't ask for "breakdown" or "by manager"

DON'T ask if ANY of these are true:
✗ Only one context exists (you checked and found 1 row)
✗ User already specified context in their question
✗ User said "total" (they want aggregate across all contexts)
✗ User asked for breakdown (they want to see all contexts separately)

**Sub-step 6c: Format clarification question**

If you need clarification, provide numbered options:

```
Template:
I found [entity name] in multiple contexts:
1. [Context 1 with SPECIFIC details: manager name, agency name]
2. [Context 2 with SPECIFIC details: manager name, agency name]  
3. Total across all contexts

Which would you like?
```

**Example:**
```json
{{
  "needs_clarification": true,
  "clarification_question": "I found Alex Smith in multiple contexts:\\n1. Under [actual manager 1] at [actual agency 1]\\n2. Under [actual manager 2] at [actual agency 2]\\n3. Total across all contexts\\n\\nWhich would you like?"
}}
```

**STEP 7: OUTPUT JSON PLAN**

Build the final JSON plan:

```json
{{
  "reasoning": "Step-by-step explanation of your thinking",
  "intent": "query|describe_schema|general|clarification_response",
  "domain": "salesperson|agent|unclear",
  "needs_db": true|false,
  "explicit_visualization": true|false,
  "tables": ["table1", "table2"],
  "joins": ["table1.col = table2.col"],
  "conditions": ["WHERE clauses"],
  "select_columns": ["columns or aggregates to select"],
  "logic_summary": "What the SQL should accomplish",
  "needs_clarification": true|false,
  "clarification_question": "Question with numbered options (if needed)",
  "resolved_context": "If user answered clarification, what context",
  "notes": "Special SQL instructions (column quoting, date functions, etc.)"
}}
```

**IMPORTANT:** Set "explicit_visualization": true if user said:
- "create a histogram", "show me a chart", "plot the data"
- "bar chart", "line graph", "visualize", etc.

════════════════════════════════════════════════════════════════════
COMPLETE EXAMPLES
════════════════════════════════════════════════════════════════════

**Example 1: Simple Query - NO training**

User: "Sales for Bob in 2024?"

Step 1: Check history → No previous clarification
Step 2: Intent → Data retrieval (no visualization keywords)
Step 3: Domain → "sales" keyword → Salesperson
Step 4: Context → Not a follow-up
Step 5: Tables → orders, salesperson (NO training - user didn't ask for "valid")
Step 6: Clarification → "Bob" probably unique → No clarification

Output:
{{
  "reasoning": "User wants total sales for Bob in 2024. General 'sales' query - do NOT use training table.",
  "intent": "query",
  "domain": "salesperson",
  "needs_db": true,
  "explicit_visualization": false,
  "tables": ["orders", "salesperson"],
  "joins": ["orders.salesperson_id = salesperson.ID"],
  "conditions": ["WHERE salesperson.Name = 'Bob'", "YEAR(order_date) = 2024"],
  "select_columns": ["SUM(orders.Amount) as total_sales"],
  "logic_summary": "Sum ALL orders for Bob in 2024 (no training filter)",
  "needs_clarification": false,
  "notes": "Use YEAR() function. DO NOT join training table."
}}

**Example 1b: Explicit Visualization Request**

User: "Create a histogram for sales"

Step 1: Check history → No clarification
Step 2: Intent → Data retrieval + Visualization ("histogram" keyword detected!)
Step 3: Domain → "sales" → Salesperson
Step 4: Context → Not a follow-up
Step 5: Tables → orders (get sales data)
Step 6: Clarification → No clarification needed

Output:
{{
  "reasoning": "User wants to create a histogram. 'Histogram' is a visualization keyword - set explicit_visualization = true.",
  "intent": "query",
  "domain": "salesperson",
  "needs_db": true,
  "explicit_visualization": true,
  "tables": ["orders"],
  "conditions": [],
  "select_columns": ["Amount", "order_date"],
  "logic_summary": "Get all sales amounts for histogram distribution",
  "needs_clarification": false,
  "notes": "User requested histogram - visualization will auto-generate"
}}

**Example 2: Needs Clarification - Multiple contexts found**

User: "Commissions for Alex Smith"

Step 1: Check history → No clarification
Step 2: Intent → Data retrieval  
Step 3: Domain → "commissions" → Agent
Step 4: Context → Not a follow-up
Step 5: Tables → agent_commissions
Step 6a: Check for multiple contexts:
  → Use the CONTEXT CHECK RESULTS injected into this prompt (real DB data)
  → If context_check shows multiple rows for this name → multiple contexts exist!

Step 6b: Need clarification?
  - Multiple contexts ✓
  - User didn't specify ✓
  - User didn't say "total" ✓
  → YES, ask clarification!

Output:
{{
  "reasoning": "Found Alex Smith in agent_commissions under 2 different managers. Need to know which context user wants.",
  "intent": "query",
  "domain": "agent",
  "needs_db": true,
  "tables": ["agent_commissions"],
  "needs_clarification": true,
  "clarification_question": "I found Alex Smith in multiple contexts:\\n1. Under [actual manager 1] at [actual agency 1]\\n2. Under [actual manager 2] at [actual agency 2]\\n3. Total across all contexts\\n\\nWhich would you like?",
  "notes": "Check actual data to verify multiple contexts"
}}

**Example 3: User Answers Clarification**

Conversation:
- User: "Commissions for Alex Smith"
- Assistant: "Which Alex? 1) Under [manager 1] 2) Under [manager 2] 3) Total"
- User: "option 1"

Step 1: Check history → Previous message has clarification with options!
  User said: "option 1"
  My option 1 was: whatever was listed first in my clarification_question
  
  Resolve: upline_manager and agency_name from that option

Step 2-5: [Build plan with resolved context]
Step 6: Clarification → Already resolved! Set needs_clarification = false

Output:
{{
  "reasoning": "User selected option 1 — resolved to the first context from CONTEXT CHECK RESULTS.",
  "intent": "clarification_response",
  "domain": "agent",
  "needs_db": true,
  "tables": ["agent_commissions"],
  "conditions": ["WHERE agent_name = 'Alex Smith'", "AND upline_manager = '[actual manager]'", "AND agency_name = '[actual agency]'"],
  "select_columns": ["SUM of all monthly commission columns as total"],
  "logic_summary": "Sum all commissions for Alex Smith under the resolved manager",
  "needs_clarification": false,
  "resolved_context": "Alex Smith under [actual manager] at [actual agency]",
  "notes": "Monthly columns like '2024-01-01' need double quotes in SQL"
}}

**Example 4: Bonus Query - USES training**

User: "What's Bob's bonus for 2024?"

Step 1: Check history → No clarification
Step 2: Intent → Data retrieval
Step 3: Domain → "bonus" → Salesperson
Step 4: Context → Not a follow-up
Step 5: Tables → Need to calculate VALID sales first!
  - "bonus" keyword → Must use training table
  - Tables: orders, salesperson, training, bonus_pay
  - Process: ONE query with CTE/subquery (not two queries!)

Step 6: Clarification → "Bob" probably unique → No clarification

Output:
{{
  "reasoning": "Bonus query - bonuses are based on VALID sales only. Must filter by training windows, then compare to bonus_pay tiers in ONE query using CTE.",
  "intent": "query",
  "domain": "salesperson",
  "needs_db": true,
  "tables": ["orders", "salesperson", "training", "bonus_pay"],
  "joins": ["orders.salesperson_id = salesperson.ID", "orders with training on salesperson_id AND date range"],
  "select_columns": ["Bob's name", "Total valid sales", "MAX(bonus) from bonus_pay"],
  "logic_summary": "Use CTE to calculate valid sales, then SELECT MAX(bonus) WHERE tier <= valid_sales. ONE query, not two!",
  "needs_clarification": false,
  "notes": "CRITICAL: Use WITH clause (CTE) or subquery. Do NOT write two separate queries. bonus_pay is lookup table - compare calculated total to tiers in single query."
}}

**Example 4b: Top 5 Bonuses (Multiple People)**

User: "Top 5 bonus earners in 2012"

Output:
{{
  "reasoning": "Need valid sales for all salespeople, then match to bonus tiers, order by bonus.",
  "intent": "query",
  "domain": "salesperson",
  "needs_db": true,
  "tables": ["orders", "salesperson", "training", "bonus_pay"],
  "select_columns": ["Name", "Total valid sales", "Earned bonus"],
  "logic_summary": "CTE calculates valid sales per person. Main query matches each total to bonus_pay tiers using correlated subquery. ORDER BY bonus DESC LIMIT 5. ONE query!",
  "needs_clarification": false,
  "notes": "Use WITH to get sales per person, then (SELECT MAX(bonus) WHERE tier <= total_valid) for each person. Single query with CTE + correlated subquery."
}}

════════════════════════════════════════════════════════════════════
CRITICAL FINAL REMINDERS
════════════════════════════════════════════════════════════════════

✓ ALWAYS check conversation_history for previous clarifications
✓ Use CONTEXT CHECK RESULTS (injected above) to verify multiple contexts — never invent them
✓ Map "option 1", "manager A" to actual choices from your previous question
✓ ONLY use training table when user asks for "valid", "invalid", or "bonus"
✓ For bonus: training is required (bonuses use valid sales)
✓ bonus_pay is a LOOKUP table (compare totals to tiers, don't join by ID)
✓ Provide numbered options when asking clarifications
✓ Never mix salesperson and agent domains
✓ Output valid JSON only

Now analyze the question using these exact steps and output your JSON plan!
"""