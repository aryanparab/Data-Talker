import agents
from langchain_core.messages import SystemMessage, HumanMessage
import agent_tools
import json
from graph_agent import generate_graph, execute_graph_code

MAX_TOOL_ROUNDS = 10

EXECUTOR_SYSTEM = agent_tools.get_execution_prompt()


def main():
    # Executor MUST be Groq — it needs bind_tools for structured tool calling
    executor_llm   = agents.load_llm("executor")
    _, tools       = agent_tools.get_tools()
    llm_with_tools = executor_llm.bind_tools(tools)
    schemas        = agents._schemas()

    # Pre-extract table names directly from the live schema
    table_names = [
        line.replace("TABLE:", "").strip()
        for line in schemas.splitlines()
        if line.startswith("TABLE:")
    ]

    if agents.USE_HUGGINGFACE_API and agents.HF_API_KEY:
        print("=" * 60)
        print("  Multi-Model DB Agent")
        print(f"  - Planner : {agents.PLANNER_MODEL} (HF API)")
        print(f"  - SQL     : {agents.SQL_MODEL} (HF API)")
        print(f"  - Executor: Groq  (tool calling)")
        print(f"  - Viz     : Groq  (on-demand)")
        print("=" * 60)
    else:
        print("─" * 50)
        print("  DB Agent (Groq).  Type 'quit' to exit.")
        print("─" * 50)

    print(f"\n  Tables available: {', '.join(table_names)}")
    print("\nType 'quit' to exit.")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if not user_input:
            continue

        conversation_history = [f"User: {user_input}"]

        for _ in range(MAX_TOOL_ROUNDS):

            print("\n  [planner] Analyzing query...")
            plan = agents.get_plan(user_input, conversation_history)
            print(f"  [planner] Tables: {plan.get('tables', [])}")
            if plan.get("logic_summary"):
                print(f"  [planner] Logic : {plan['logic_summary']}")

            # ── No DB query needed ─────────────────────────────────────
            if not plan.get("needs_db", True):
                response = executor_llm.invoke([
                    SystemMessage(content="You are a helpful data assistant."),
                    HumanMessage(content=user_input),
                ])
                print(f"\nAgent: {response.content}")
                break

            # ── Describe schema — call get_schema live from DB ─────────
            if plan.get("intent") == "describe_schema":
                TOOL_MAP, _ = agent_tools.get_tools()
                tables_to_describe = plan.get("tables") or table_names

                schema_parts = []
                for tbl in tables_to_describe:
                    print(f"  [schema] Fetching {tbl}...")
                    schema_info = TOOL_MAP["get_schema"].invoke({"table_name": tbl})
                    schema_parts.append(f"### {tbl}\n{schema_info}")

                combined = "\n\n".join(schema_parts)

                # Ask executor LLM to write a human-friendly summary
                summary = executor_llm.invoke([
                    SystemMessage(content=(
                        "You are a helpful data assistant. "
                        "Given the raw schema and sample data for each table, "
                        "write a clear, friendly description of what each table contains "
                        "and what kinds of questions can be answered with it."
                    )),
                    HumanMessage(content=(
                        f"User asked: {user_input}\n\n"
                        f"Here is the live schema info:\n\n{combined}"
                    )),
                ])
                print(f"\nAgent: {summary.content}")
                break

            # ── Clarification needed ───────────────────────────────────
            if plan.get("needs_clarification"):
                clarification = plan["clarification_question"]
                print(f"\nAgent: {clarification}")
                conversation_history.append(f"Agent: {clarification}")
                user_response = input("\nYou: ").strip()
                if user_response.lower() in ("quit", "exit", "q"):
                    print("Bye.")
                    return
                conversation_history.append(f"User: {user_response}")
                continue

            # ── Execute query ──────────────────────────────────────────
            query_result   = None
            sql_succeeded  = False

            # Try HF specialized SQL model first
            if hasattr(agents, "generate_sql_with_specialized_model"):
                success, result, sql = agents.generate_sql_with_specialized_model(
                    plan, user_input
                )
                if success:
                    query_result  = result
                    sql_succeeded = True
                    answer = executor_llm.invoke([
                        SystemMessage(content=(
                            "You are a helpful data assistant. "
                            "Answer clearly and concisely in plain language."
                        )),
                        HumanMessage(content=(
                            f"Question: {user_input}\n"
                            f"SQL used:\n{sql}\n\n"
                            f"Result:\n{result}\n\n"
                            "Answer the question directly."
                        )),
                    ])
                    print(f"\nAgent: {answer.content}")
                    print("\nQuery Completed!")

            # ── Fallback: Groq with structured tool calls ──────────────
            if not sql_succeeded:
                print("\n  [executor] Running with Groq tool-calling...")

                executor_messages = [
                    SystemMessage(content=EXECUTOR_SYSTEM),
                    HumanMessage(content=(
                        f"=== SCHEMAS ===\n{schemas}\n\n"
                        f"=== QUERY PLAN ===\n{json.dumps(plan, indent=2)}\n\n"
                        f"=== CONVERSATION HISTORY ===\n{' '.join(conversation_history)}\n\n"
                        f"=== USER QUESTION ===\n{user_input}\n\n"
                        "Write and run the SQL now using the run_query tool."
                    )),
                ]

                for _ in range(MAX_TOOL_ROUNDS):
                    response = llm_with_tools.invoke(executor_messages)
                    executor_messages.append(response)

                    done, tool_messages = agents.handle_response(response)

                    if tool_messages:
                        for msg in tool_messages:
                            if not msg.content.startswith("ERROR"):
                                query_result = msg.content

                    if done:
                        print(f"\nAgent: {response.content}")
                        print("\nQuery Completed!")
                        break

                    executor_messages.extend(tool_messages)
                else:
                    print("\nAgent: ⚠️  Too many steps — please rephrase your question.")

            # ── Optional visualization ─────────────────────────────────
            wants_graph = (
                any(w in user_input.lower() for w in
                    ["chart", "graph", "plot", "visualize", "show"])
                or plan.get("visualization_recommended", False)
            )

            if wants_graph and query_result:
                print("\n  [graph agent] Generating visualization...")
                graph_result = generate_graph(query_result, user_input)
                if not graph_result["success"]:
                    print(f"  [graph agent] ❌ {graph_result['error']}")
                else:
                    exec_result = execute_graph_code(graph_result["code"], "output_chart.png")
                    if exec_result["success"]:
                        print(f"  [graph agent] ✅ Chart saved: {exec_result['output_path']}")
                    else:
                        print(f"  [graph agent] ❌ {exec_result['error']}")

            break  # exit clarification loop after execution


if __name__ == "__main__":
    main()