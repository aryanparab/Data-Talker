import agents
from langchain_core.messages import SystemMessage, HumanMessage
import agent_tools
import json
from graph_agent import generate_graph, execute_graph_code
import duckdb


MAX_TOOL_ROUNDS = 10
DB_PATH = "salesdata.db"

EXECUTOR_SYSTEM = agent_tools.get_execution_prompt()

REFERENCE_QUERIES = {
    "show all salespeople": """SELECT * FROM salesperson""",
    
    "how many orders are there": """SELECT COUNT(*) as total_orders FROM orders""",
    
    "show bob's details": """SELECT * FROM salesperson WHERE name = 'Bob'""",
    
    "show orders with salesperson names": """
        SELECT o.id, o.order_date, o.amount, s.name as salesperson_name
        FROM orders o
        JOIN salesperson s ON o.salesperson_id = s.id
    """,
    
    "what is the total amount of all orders": """SELECT SUM(amount) as total_revenue FROM orders""",
    
    "show total sales per salesperson": """
        SELECT s.name, SUM(o.amount) as total_sales
        FROM salesperson s
        JOIN orders o ON s.id = o.salesperson_id
        GROUP BY s.name
        ORDER BY total_sales DESC
    """,
    
    "find all valid orders": """
        SELECT o.id, o.order_date, o.salesperson_id, o.amount, s.name
        FROM orders o
        JOIN salesperson s ON o.salesperson_id = s.id
        JOIN training t ON o.salesperson_id = t.salesperson_id
        WHERE o.order_date BETWEEN t.start_date AND t.end_date
    """,
    
    "find all invalid orders": """
        SELECT o.id, o.order_date, o.salesperson_id, o.amount, s.name
        FROM orders o
        JOIN salesperson s ON o.salesperson_id = s.id
        WHERE NOT EXISTS (
            SELECT 1 FROM training t
            WHERE t.salesperson_id = o.salesperson_id
            AND o.order_date BETWEEN t.start_date AND t.end_date
        )
    """,
    
    "how much did bob sell in 2012": """
        SELECT SUM(o.amount) as total_valid_sales
        FROM orders o
        JOIN salesperson s ON o.salesperson_id = s.id
        JOIN training t ON o.salesperson_id = t.salesperson_id
        WHERE s.name = 'Bob'
        AND EXTRACT(YEAR FROM o.order_date) = 2012
        AND o.order_date BETWEEN t.start_date AND t.end_date
    """,
    
    "which agents have multiple upline managers": """
        SELECT agent_name, COUNT(DISTINCT upline_manager) as manager_count
        FROM agent_commissions
        GROUP BY agent_name
        HAVING COUNT(DISTINCT upline_manager) > 1
    """,
}


def find_reference_query(user_input):
    """Find matching reference query if exists"""
    user_lower = user_input.lower().strip()
    
    # Exact match
    if user_lower in REFERENCE_QUERIES:
        return REFERENCE_QUERIES[user_lower]
    
    # Partial match (fuzzy)
    for key, sql in REFERENCE_QUERIES.items():
        if key in user_lower or user_lower in key:
            return sql
    
    return None


def extract_sql_from_messages(messages):
    """Extract SQL query from executor messages"""
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get('name') == 'run_query':
                    return tool_call.get('args', {}).get('sql')
    return None


def verify_sql(llm_sql, reference_sql):
    """Compare LLM SQL vs reference SQL results"""
    print("\n" + "="*80)
    print("📊 SQL VERIFICATION")
    print("="*80)
    
    conn = duckdb.connect(DB_PATH)
    
    # Execute LLM query
    print("\n🤖 LLM-Generated SQL:")
    print(llm_sql)
    
    try:
        llm_result = conn.execute(llm_sql).fetchdf()
        llm_success = True
        print(f"✓ Executed successfully ({len(llm_result)} rows)")
    except Exception as e:
        llm_result = None
        llm_success = False
        print(f"✗ Failed: {e}")
    
    # Execute reference query
    print("\n📋 Reference SQL:")
    print(reference_sql.strip())
    
    try:
        ref_result = conn.execute(reference_sql).fetchdf()
        ref_success = True
        print(f"✓ Executed successfully ({len(ref_result)} rows)")
    except Exception as e:
        ref_result = None
        ref_success = False
        print(f"✗ Failed: {e}")
    
    conn.close()
    
    # Compare results
    print("\n" + "─"*80)
    print("COMPARISON")
    print("─"*80)
    
    if not llm_success:
        print("❌ FAILED - LLM query has errors")
        return False
    
    if not ref_success:
        print("⚠️  Reference query error")
        return False
    
    # Check exact match
    if llm_result.equals(ref_result):
        print("✅ PERFECT MATCH!")
        print(f"\n📊 Results ({len(llm_result)} rows):")
        print(llm_result.to_string(index=False))
        print("\n" + "="*80 + "\n")
        return True
    
    # Try sorted comparison
    try:
        llm_sorted = llm_result.sort_values(by=list(llm_result.columns)).reset_index(drop=True)
        ref_sorted = ref_result.sort_values(by=list(ref_result.columns)).reset_index(drop=True)
        
        if llm_sorted.equals(ref_sorted):
            print("✅ MATCH (after sorting)!")
            print(f"\n📊 Results ({len(llm_result)} rows):")
            print(llm_result.to_string(index=False))
            print("\n" + "="*80 + "\n")
            return True
    except:
        pass
    
    # Results differ
    print("❌ MISMATCH - Results are different!")
    print(f"\n🤖 LLM Result ({len(llm_result)} rows):")
    print(llm_result.to_string(index=False))
    print(f"\n📋 Expected ({len(ref_result)} rows):")
    print(ref_result.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    return False


def main():
    llm            = agents.load_llm()
    _, tools = agent_tools.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    schemas        = agents._schemas()                 

    print("─" * 80)
    print("  DB Agent (Planner + Executor) with SQL Verification")
    print("  Commands: 'quit' - exit | 'verify on/off' - toggle verification")
    print("─" * 80)
    
    # Verification mode (toggle with 'verify on' or 'verify off')
    verification_enabled = True
    print(f"\n✓ SQL Verification: {'ENABLED' if verification_enabled else 'DISABLED'}")
    print("  (Compares LLM SQL vs reference when available)\n")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        
        if user_input.lower() == "verify on":
            verification_enabled = True
            print("✓ SQL Verification ENABLED")
            continue
        
        if user_input.lower() == "verify off":
            verification_enabled = False
            print("✓ SQL Verification DISABLED")
            continue
        
        if not user_input:
            continue

        # Check for reference query
        reference_sql = find_reference_query(user_input) if verification_enabled else None
        if reference_sql:
            print("📌 Reference query found - will verify LLM output")

        # Start conversation history for this question
        conversation_history = [f"User: {user_input}"]
        
        # Clarification loop
        for _ in range(MAX_TOOL_ROUNDS):
            
            print("\n  [planner] thinking …")
            plan = agents.get_plan(user_input, conversation_history)
            print(f"  [planner] done →  tables={plan.get('tables')}")
            if plan.get("logic_summary"):
                print(f"            logic: {plan['logic_summary']}")

            # No DB needed
            if not plan.get("needs_db", True):
                response = llm.invoke([
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=user_input),
                ])
                print(f"\nAgent: {response.content}")
                break

            # Needs clarification
            if plan.get("needs_clarification"):
                clarification = plan['clarification_question']
                print(f"\nAgent: {clarification}")
                conversation_history.append(f"Agent: {clarification}")
                
                # Get clarification from user
                user_response = input("\nYou: ").strip()
                if user_response.lower() in ("quit", "exit", "q"):
                    print("Bye.")
                    return
                conversation_history.append(f"User: {user_response}")
                continue

            # Execute query
            executor_messages = [
                SystemMessage(content=EXECUTOR_SYSTEM),
                HumanMessage(content=(
                    f"=== SCHEMAS ===\n{schemas}\n\n"
                    f"=== QUERY PLAN ===\n{json.dumps(plan, indent=2)}\n\n"
                    f"=== CONVERSATION HISTORY ===\n{' '.join(conversation_history)}\n\n"
                    f"=== USER QUESTION ===\n{user_input}\n\n"
                    "Now write and run the SQL."
                )),
            ]

            # Tool call loop
            query_result = None
            llm_sql = None
            
            for _ in range(MAX_TOOL_ROUNDS):
                response = llm_with_tools.invoke(executor_messages)
                executor_messages.append(response)
                
                # Extract SQL from this response
                if not llm_sql:
                    llm_sql = extract_sql_from_messages([response])

                done, tool_messages = agents.handle_response(response)
                
                if tool_messages:
                    for msg in tool_messages:
                        if not msg.content.startswith("ERROR"):
                            query_result = msg.content
                
                if done:
                    print(f"\nAgent: {response.content}")
                    print("\n✓ Query Completed")
                    
                    # ═══════════════════════════════════════
                    # SQL VERIFICATION STEP
                    # ═══════════════════════════════════════
                    if verification_enabled and reference_sql and llm_sql:
                        verify_sql(llm_sql, reference_sql)
                    
                    break

                executor_messages.extend(tool_messages)
            else:
                print("\nAgent: [stopped — too many tool calls, please rephrase]")
            
            # Visualization
            wants_graph = (
                any(word in user_input.lower() for word in 
                    ['chart', 'graph', 'plot', 'visualize', 'show']) 
                or plan.get("visualization_recommended", False)
            )
        
            if wants_graph and query_result:
                print("\n  [graph agent] generating visualization...")
                
                graph_result = generate_graph(query_result, user_input)
                
                if not graph_result["success"]:
                    print(f"  [graph agent] ❌ Failed: {graph_result['error']}")
                else:
                    print(f"  [graph agent] Chart type: {graph_result['chart_type']}")
                    print(f"  [graph agent] {graph_result['reasoning']}")
                    
                    exec_result = execute_graph_code(graph_result['code'], "output_chart.png")
                    
                    if exec_result["success"]:
                        print(f"  [graph agent] ✓ Chart saved to: {exec_result['output_path']}")
                    else:
                        print(f"  [graph agent] ❌ Execution failed: {exec_result['error']}")
            
            break


if __name__ == "__main__":
    main()