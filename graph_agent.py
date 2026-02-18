# graph_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

GRAPH_SYSTEM_PROMPT = """\
You are a data visualization expert using conversation context to create meaningful charts.

You will receive:
1. SQL query results (as a DataFrame string)
2. User's original question/request
3. Conversation context (previous questions if relevant)

DETERMINE VISUALIZATION NEED:
- **Explicit request**: User said "chart", "graph", "plot", "visualize", "show graph"
- **Implicit**: Button clicked after query - user wants to see the data visually

Your task:
- Detect if this was an explicit visualization request
- Analyze data structure and choose appropriate chart type
- Use conversation context to build meaningful title
- Write clean, executable Python code

CHART TYPE SELECTION:
- **Bar**: Categorical comparisons (sales by person, counts by category)
- **Horizontal bar**: Rankings (top N items)
- **Line**: Time series, trends (sales over months/years)
- **Scatter**: Two continuous variables relationship
- **Pie**: Parts of whole (use only if <7 categories)
- **Grouped bar**: Multiple metrics across categories
- **Heatmap**: Matrix data, correlations

CRITICAL REQUIREMENTS:
- Parse data string into DataFrame properly (handle whitespace-separated values)
- Add descriptive title based on user's actual question
- Use readable fonts (10-12pt)
- Add proper axis labels
- Handle empty/invalid data gracefully
- Save: plt.savefig('output.png', bbox_inches='tight', dpi=300)
- Close: plt.close()

CONTEXT AWARENESS EXAMPLES:
```
Previous: "Sales for Bob in 2024?"
Current: "Now for Alice"
→ Title: "Sales for Alice in 2024"

Previous: "Top 5 agents by commission?"  
Button clicked
→ Title: "Top 5 Agents by Commission"
```

OUTPUT FORMAT (Valid JSON):
{
  "explicit_request": true|false,
  "chart_type": "bar|line|scatter|pie|...",
  "reasoning": "Why this chart type fits the data",
  "title": "Chart title from user's question",
  "code": "Python code with \\n for newlines"
}

EXAMPLE:
{
  "explicit_request": false,
  "chart_type": "bar",
  "reasoning": "Data has 5 people with sales values. Bar chart best for comparing across categories.",
  "title": "Sales by Salesperson",
  "code": "import pandas as pd\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\nimport io\\ndata = '''name sales\\nBob 1500\\nAlice 2300\\nJoe 1800'''\\ndf = pd.read_csv(io.StringIO(data), sep='\\\\s+')\\nplt.figure(figsize=(10,6))\\nsns.barplot(data=df, x='name', y='sales')\\nplt.title('Sales by Salesperson', fontsize=14)\\nplt.xlabel('Name')\\nplt.ylabel('Sales ($)')\\nplt.tight_layout()\\nplt.savefig('output.png', bbox_inches='tight', dpi=300)\\nplt.close()"
}
"""


def load_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0,
    )


def clean_json_response(raw: str) -> dict:
    """
    Clean up malformed JSON responses from LLM.
    Handles both escaped newlines (\\n) and actual newlines in JSON.
    Also handles escaped field names and new fields: explicit_request, title.
    """
    # Remove markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        start = 1
        if lines[0].startswith("```json"):
            start = 1
        end = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        raw = "\n".join(lines[start:end])
    
    # Clean up escaped quotes in field names (common LLM error)
    raw = raw.replace('\\"explicit_request\\"', '"explicit_request"')
    raw = raw.replace('\\"chart_type\\"', '"chart_type"')
    raw = raw.replace('\\"reasoning\\"', '"reasoning"')
    raw = raw.replace('\\"title\\"', '"title"')
    raw = raw.replace('\\"code\\"', '"code"')
    
    # Try to parse as-is first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Strategy: Extract fields manually
    try:
        result = {}
        
        # Extract explicit_request (boolean)
        explicit_match = re.search(r'["\']?explicit_request["\']?\s*:\s*(true|false)', raw)
        result["explicit_request"] = explicit_match.group(1) == "true" if explicit_match else False
        
        # Extract chart_type
        chart_type_match = re.search(r'["\']?chart_type["\']?\s*:\s*["\']([^"\']+)["\']', raw)
        result["chart_type"] = chart_type_match.group(1) if chart_type_match else "unknown"
        
        # Extract reasoning
        reasoning_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']\s*[,}]', raw, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).replace('\\"', '"')
        else:
            result["reasoning"] = "No reasoning provided"
        
        # Extract title
        title_match = re.search(r'["\']?title["\']?\s*:\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']\s*[,}]', raw, re.DOTALL)
        if title_match:
            result["title"] = title_match.group(1).replace('\\"', '"')
        else:
            result["title"] = "Data Visualization"
        
        # Extract code (same complex logic as before)
        code_pattern = r'["\']?code["\']?\s*:\s*["\']'
        code_match = re.search(code_pattern, raw)
        
        if not code_match:
            raise ValueError("Could not find 'code' field")
        
        code_start_pos = code_match.end()
        code_chars = []
        i = code_start_pos
        escape_next = False
        
        while i < len(raw):
            char = raw[i]
            
            if escape_next:
                if char == 'n':
                    code_chars.append('\n')
                elif char == 't':
                    code_chars.append('\t')
                elif char == 'r':
                    code_chars.append('\r')
                elif char == '"' or char == "'":
                    code_chars.append(char)
                elif char == '\\':
                    code_chars.append('\\')
                else:
                    code_chars.append('\\')
                    code_chars.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                i += 1
                continue
            
            if char == '"' or char == "'":
                remaining = raw[i+1:].lstrip()
                if remaining.startswith('}') or remaining.startswith(','):
                    result["code"] = ''.join(code_chars)
                    return result
                else:
                    code_chars.append(char)
                    i += 1
                    continue
            
            code_chars.append(char)
            i += 1
        
        if code_chars:
            result["code"] = ''.join(code_chars)
            return result
        else:
            raise ValueError("Could not extract code field content")
    
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw:\n{raw[:500]}...")


def generate_graph(data_result: str, user_request: str, conversation_context: list = None) -> dict:
    """
    Takes query results, user's request, and conversation context.
    Returns a dict with chart_type, reasoning, and executable code.
    
    Args:
        data_result: String representation of query results (DataFrame.to_string())
        user_request: User's original question or chart request
        conversation_context: List of recent conversation messages (optional)
    
    Returns:
        {
            "explicit_request": bool,
            "chart_type": str,
            "reasoning": str,
            "title": str,
            "code": str,
            "success": bool,
            "error": str (if failed)
        }
    """
    llm = load_llm()
    
    # Build context string if provided
    context_str = ""
    if conversation_context and len(conversation_context) > 0:
        context_str = "\n\nCONVERSATION CONTEXT (last few messages):\n"
        for msg in conversation_context[-5:]:  # Last 5 messages
            context_str += f"{msg}\n"
    
    prompt = f"""\
USER REQUEST:
{user_request}
{context_str}
QUERY RESULTS:
{data_result}

Analyze the request and data, then generate visualization code as JSON.
"""
    
    messages = [
        SystemMessage(content=GRAPH_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    raw = response.content.strip()
    
    try:
        result = clean_json_response(raw)
        result["success"] = True
        result["error"] = None
        
        # Ensure all expected fields exist
        result.setdefault("explicit_request", False)
        result.setdefault("title", "Data Visualization")
        
        return result
    except Exception as e:
        return {
            "explicit_request": False,
            "chart_type": "unknown",
            "reasoning": "Failed to parse response",
            "title": "Visualization",
            "code": "",
            "success": False,
            "error": f"Parse error: {e}\nRaw response:\n{raw[:500]}..."
        }


def execute_graph_code(code: str, output_path: str = "chart.png") -> dict:
    """
    Executes the generated Python code in a safe environment.
    
    Args:
        code: Python code string to execute
        output_path: Where to save the chart
    
    Returns:
        {
            "success": bool,
            "output_path": str,
            "error": str (if failed)
        }
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Prepare safe execution environment
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import io
    from datetime import datetime
    
    safe_globals = {
        '__builtins__': __builtins__,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'np': np,
        "io": io,
        "datetime": datetime,
    }
    
    # Modify code to use our output path
    code = code.replace("'output.png'", f"'{output_path}'")
    code = code.replace('"output.png"', f'"{output_path}"')
    
    try:
        exec(code, safe_globals)
        return {
            "success": True,
            "output_path": output_path,
            "error": None
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "output_path": None,
            "error": f"Execution error: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }


# Example usage / testing function
def test_graph_agent():
    """Test the graph agent with sample data"""
    
    # Sample data (what would come from SQL query)
    sample_data = """
   name  total_sales  num_orders
    Joe        700.0           1
    Tom       2200.0           1
   Josh       2700.0           2
  Alice       1600.0           1
    Bob       1640.0           2
"""
    
    user_request = "Create a bar chart showing total sales by salesperson"
    
    print("Generating visualization code...")
    result = generate_graph(sample_data, user_request)
    
    if not result["success"]:
        print(f"❌ Failed: {result['error']}")
        return
    
    print(f"✓ Chart type: {result['chart_type']}")
    print(f"✓ Reasoning: {result['reasoning']}")
    print(f"\nGenerated code:\n{'-'*50}")
    print(result['code'])
    print('-'*50)
    
    print("\nExecuting code...")
    exec_result = execute_graph_code(result['code'], "test_chart.png")
    
    if exec_result["success"]:
        print(f"✓ Chart saved to: {exec_result['output_path']}")
    else:
        print(f"❌ Execution failed: {exec_result['error']}")


if __name__ == "__main__":
    test_graph_agent()