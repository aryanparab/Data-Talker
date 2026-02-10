# graph_agent.py

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

GRAPH_SYSTEM_PROMPT = """\
You are a data visualization expert. Your job is to create Python code that generates charts using matplotlib and/or seaborn.

You will receive:
1. SQL query results (as a DataFrame string)
2. User's visualization request

Your task:
- Analyze the data structure
- Choose the appropriate chart type (bar, line, scatter, pie, heatmap, etc.)
- Write clean, executable Python code
- Use matplotlib/seaborn for visualization
- Save the figure to a file

CRITICAL REQUIREMENTS:
- Code must be complete and executable
- Import all necessary libraries
- Use the data provided (convert string to DataFrame if needed)
- Save figure using plt.savefig('output.png', bbox_inches='tight', dpi=300)
- Close the plot with plt.close()
- Add proper labels, title, and legend
- Handle edge cases (empty data, single values, etc.)

AVAILABLE LIBRARIES:
- pandas as pd
- matplotlib.pyplot as plt
- seaborn as sns
- numpy as np

OUTPUT FORMAT:
You MUST respond with a valid JSON object. For the code field, escape all newlines and quotes properly.

Structure:
{
  "chart_type": "bar|line|scatter|pie|heatmap|...",
  "reasoning": "Why this chart type is appropriate",
  "code": "complete Python code on a SINGLE LINE with \\n for newlines"
}

IMPORTANT: The "code" field must be a single string with \\n for line breaks, not actual newlines.

EXAMPLE:
{
  "chart_type": "bar",
  "reasoning": "Bar chart shows comparison clearly",
  "code": "import pandas as pd\\nimport matplotlib.pyplot as plt\\ndf = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})\\nplt.bar(df['x'], df['y'])\\nplt.savefig('output.png', bbox_inches='tight', dpi=300)\\nplt.close()"
}
"""


def load_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0,
    )


def clean_json_response(raw: str) -> str:
    """
    Clean up malformed JSON responses from LLM.
    Handles multiline code blocks in JSON.
    """
    # Remove markdown fences
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
    
    # Try to parse as-is first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Fallback: manually extract and fix the code field
    # Find the code field and escape it properly
    try:
        # Extract chart_type
        chart_type_match = re.search(r'"chart_type"\s*:\s*"([^"]+)"', raw)
        chart_type = chart_type_match.group(1) if chart_type_match else "unknown"
        
        # Extract reasoning
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+(?:\\.[^"]*)*)"', raw)
        reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
        
        # Extract code (everything between "code": " and the last ")
        code_match = re.search(r'"code"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            # The code might have actual newlines, keep them as-is
        else:
            # Try alternate pattern - multiline code block
            code_match = re.search(r'"code"\s*:\s*"([^"]*)$', raw, re.DOTALL)
            if code_match:
                # Extract everything after "code": "
                remaining = raw[code_match.end():]
                # Find the closing "
                code_end = remaining.rfind('"')
                if code_end != -1:
                    code = remaining[:code_end]
                else:
                    raise ValueError("Could not parse code field")
            else:
                raise ValueError("Could not find code field")
        
        return {
            "chart_type": chart_type,
            "reasoning": reasoning,
            "code": code
        }
    
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw:\n{raw}")


def generate_graph(data_result: str, user_request: str) -> dict:
    """
    Takes query results and user's visualization request.
    Returns a dict with chart_type, reasoning, and executable code.
    
    Args:
        data_result: String representation of query results (DataFrame.to_string())
        user_request: User's original question or chart request
    
    Returns:
        {
            "chart_type": str,
            "reasoning": str,
            "code": str,
            "success": bool,
            "error": str (if failed)
        }
    """
    llm = load_llm()
    
    prompt = f"""\
USER REQUEST:
{user_request}

QUERY RESULTS:
{data_result}

Now generate the visualization code. Remember: output valid JSON with escaped newlines in the code field.
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
        return result
    except Exception as e:
        return {
            "chart_type": "unknown",
            "reasoning": "Failed to parse response",
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
    
    safe_globals = {
        '__builtins__': __builtins__,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'np': np,
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