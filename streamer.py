import streamlit as st
import agents
import agent_tools
import json
from graph_agent import generate_graph, execute_graph_code
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DB Query Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS = 10
EXECUTOR_SYSTEM = agent_tools.get_execution_prompt()
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Pre-flight context check ─────────────────────────────────────────────────
def get_agent_context(user_input: str, tool_map: dict, llm) -> str:
    """
    Before planning, extract any agent/person name from the question and run
    a real DB lookup to find their upline managers and agencies.
    This result is injected into the planner prompt so it never has to guess.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        name_response = llm.invoke([
            SystemMessage(content=(
                "Extract a person or agent name from this question. "
                "Return only the name with no extra text. "
                "If there is no specific person or agent name, return NONE."
            )),
            HumanMessage(content=user_input),
        ])
        name = name_response.content.strip()

        if not name or name.upper() == "NONE" or len(name) > 60:
            return ""

        # Query both tables — agent_commissions for DS-2, salesperson for DS-1
        result = tool_map["run_query"].invoke({
            "sql": f"""
                SELECT DISTINCT upline_manager, agency_name
                FROM agent_commissions
                WHERE agent_name = '{name}'
            """
        })

        if result.startswith("ERROR") or not result.strip():
            return ""

        lines = [l for l in result.strip().split("\n") if l.strip()]
        # Only inject if multiple contexts exist (more than header + 1 data row)
        if len(lines) <= 2:
            return ""

        return f"Agent '{name}' exists in multiple contexts:\n{result}\n"

    except Exception as e:
        print(f"  [context check] Error: {e}")
        return ""


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "waiting_for_clarification" not in st.session_state:
    st.session_state.waiting_for_clarification = False

if "current_plan" not in st.session_state:
    st.session_state.current_plan = None

# Load models once — executor is always Groq (needs bind_tools)
if "executor_llm" not in st.session_state:
    st.session_state.executor_llm   = agents.load_llm("executor")
    _, st.session_state.tools       = agent_tools.get_tools()
    st.session_state.llm_with_tools = st.session_state.executor_llm.bind_tools(
        st.session_state.tools
    )
    st.session_state.schemas  = agents._schemas()
    st.session_state.tool_map = agent_tools.get_tools()[0]

    # Pre-extract table names from live schema
    st.session_state.table_names = [
        line.replace("TABLE:", "").strip()
        for line in st.session_state.schemas.splitlines()
        if line.startswith("TABLE:")
    ]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 DB Query Assistant")

    if agents.USE_HUGGINGFACE_API and agents.HF_API_KEY:
        st.success("🟢 Multi-Model Active")
        st.caption(f"Planner: {agents.PLANNER_MODEL}")
        st.caption("Executor: Groq")
    else:
        st.info("🔵 Groq Only Mode")

    st.markdown("---")
    st.markdown("### Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            for msg in st.session_state.messages:
                if "graph_path" in msg and os.path.exists(msg["graph_path"]):
                    os.remove(msg["graph_path"])
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.waiting_for_clarification = False
            st.session_state.current_plan = None
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Visualize Latest button — only shown when there is a result without a chart
    latest_result_idx = None
    for idx in range(len(st.session_state.messages) - 1, -1, -1):
        msg = st.session_state.messages[idx]
        if msg["role"] == "assistant" and "query_result" in msg and msg["query_result"]:
            latest_result_idx = idx
            break

    if latest_result_idx is not None:
        latest_msg = st.session_state.messages[latest_result_idx]
        has_graph  = "graph_path" in latest_msg

        if has_graph:
            st.button("📊 Visualize Latest", use_container_width=True, disabled=True,
                      help="Latest result already has a visualization")
        else:
            if st.button("📊 Visualize Latest", use_container_width=True,
                         help="Generate chart for most recent query"):
                with st.spinner("Creating visualization..."):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    graph_path = f"{CHART_DIR}/chart_{timestamp}.png"

                    conversation_context = []
                    for msg in st.session_state.messages[-5:]:
                        if msg["role"] == "user":
                            conversation_context.append(f"User: {msg['content']}")
                        elif msg["role"] == "assistant":
                            conversation_context.append(f"Assistant: {msg['content'][:100]}...")

                    graph_result = generate_graph(
                        latest_msg["query_result"],
                        latest_msg.get("user_question", "Visualize this data"),
                        conversation_context=conversation_context,
                    )

                    if graph_result["success"]:
                        exec_result = execute_graph_code(graph_result["code"], graph_path)
                        if exec_result["success"]:
                            st.session_state.messages[latest_result_idx]["graph_path"] = graph_path
                            st.success("✅ Visualization created!")
                            st.rerun()
                        else:
                            st.error(f"❌ Execution failed: {exec_result['error']}")
                    else:
                        st.error(f"❌ Could not generate chart: {graph_result['error']}")

    st.markdown("---")
    st.markdown("### 📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        chart_count = sum(1 for m in st.session_state.messages if "graph_path" in m)
        st.metric("Charts", chart_count)

    st.markdown("---")
    st.markdown("### 🗄️ Tables Available")
    for t in st.session_state.table_names:
        st.markdown(f"- `{t}`")

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "Show me agents ranked by total commissions",
        "Describe all the tables",
        "What data do you have?",
        "Create a bar chart of top 5 agents",
        "How many orders in 2024?",
    ]
    for example in examples:
        if st.button(f"📝 {example}", key=example, use_container_width=True):
            st.session_state.example_query = example

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.info(
            "Natural language database assistant with smart query planning, "
            "clarification handling, schema exploration, and auto-visualizations."
        )

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("💬 Database Chat Assistant")
st.caption("Ask questions about your database in natural language")

# Render all past messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render saved SQL query
        if "executed_sql" in message:
            with st.expander("🔍 SQL Executed", expanded=False):
                st.code(message["executed_sql"], language="sql")

        # Render saved chart
        if "graph_path" in message and os.path.exists(message["graph_path"]):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.image(message["graph_path"], use_container_width=True)
            with col2:
                with open(message["graph_path"], "rb") as f:
                    st.download_button(
                        label="📥 Download",
                        data=f,
                        file_name=f"chart_{idx}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{idx}",
                    )

# ── Input handling ────────────────────────────────────────────────────────────
if "example_query" in st.session_state:
    prompt = st.session_state.pop("example_query")
else:
    prompt = st.chat_input("Ask a question about your database...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Build conversation history ────────────────────────────────────────────
    if st.session_state.waiting_for_clarification:
        st.session_state.conversation_history.append(f"User: {prompt}")
        st.session_state.waiting_for_clarification = False
        user_input = st.session_state.get("clarification_original_question", prompt)
        st.session_state.pop("clarification_original_question", None)
    else:
        st.session_state.conversation_history.append(f"User: {prompt}")
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        user_input = prompt

    # ── Agent loop ────────────────────────────────────────────────────────────
    for clarification_round in range(MAX_TOOL_ROUNDS):

        with st.chat_message("assistant"):
            status_box     = st.container()
            resp_holder    = st.empty()
            query_result   = None
            executed_sql   = None
            final_response = ""

            # ── Planning ──────────────────────────────────────────────────
            with status_box:
                with st.status("🧠 Planning query...", expanded=True) as status:
                    st.write("📋 Analyzing question...")
                    # Run pre-flight context check before planning
                    context_check = get_agent_context(
                        user_input,
                        st.session_state.tool_map,
                        st.session_state.executor_llm,
                    )
                    plan = agents.get_plan(user_input, st.session_state.conversation_history, context_check)
                    st.write(f"✅ Tables: {', '.join(plan.get('tables', [])) or 'none'}")
                    with st.expander("🔍 Query Plan"):
                        st.json(plan)
                    status.update(label="✅ Plan ready", state="complete", expanded=False)

            # ── No DB needed (chitchat / greetings) ───────────────────────
            if not plan.get("needs_db", True):
                with status_box:
                    with st.status("💭 Responding...", expanded=False):
                        chitchat_response = st.session_state.executor_llm.invoke([
                            SystemMessage(content="You are a helpful data assistant."),
                            HumanMessage(content=user_input),
                        ])
                        final_response = chitchat_response.content
                resp_holder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                break

            # ── Describe schema ────────────────────────────────────────────
            if plan.get("intent") == "describe_schema":
                tables_to_describe = plan.get("tables") or st.session_state.table_names
                schema_parts = []

                with status_box:
                    with st.status("🔍 Fetching live schema...", expanded=True) as status:
                        for tbl in tables_to_describe:
                            st.write(f"  📄 Reading `{tbl}`...")
                            schema_info = st.session_state.tool_map["get_schema"].invoke(
                                {"table_name": tbl}
                            )
                            schema_parts.append(f"### {tbl}\n{schema_info}")
                        status.update(label="✅ Schema loaded", state="complete", expanded=False)

                combined = "\n\n".join(schema_parts)

                with status_box:
                    with st.status("✍️ Summarizing...", expanded=False):
                        summary = st.session_state.executor_llm.invoke([
                            SystemMessage(content=(
                                "You are a helpful data assistant. "
                                "Given raw schema and sample data, write a clear friendly description "
                                "of what each table contains and what questions can be answered."
                            )),
                            HumanMessage(content=f"User asked: {user_input}\n\nLive schema info:\n\n{combined}"),
                        ])
                        final_response = summary.content

                resp_holder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                break

            # ── Clarification needed ───────────────────────────────────────
            if plan.get("needs_clarification"):
                clarification  = plan["clarification_question"]
                final_response = f"❓ **Clarification needed:**\n\n{clarification}"
                resp_holder.markdown(final_response)
                st.session_state.conversation_history.append(f"Agent: {clarification}")
                st.session_state.waiting_for_clarification = True
                st.session_state.current_plan = plan
                st.session_state.clarification_original_question = user_input
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                break

            # ── Execute query with Groq agentic executor ───────────────────
            with status_box:
                with st.status("⚙️ Executing with Groq...", expanded=True) as status:
                    st.write("🔨 Building SQL query...")

                    # Send only schemas for tables in the plan to reduce tokens
                    plan_tables = plan.get("tables", [])
                    if plan_tables:
                        all_blocks = st.session_state.schemas.split("TABLE:")
                        relevant_blocks = [
                            "TABLE:" + block for block in all_blocks
                            if any(t in block.split("\n")[0] for t in plan_tables)
                        ]
                        relevant_schema = "\n".join(relevant_blocks) if relevant_blocks else st.session_state.schemas
                    else:
                        relevant_schema = st.session_state.schemas

                    executor_messages = [
                        SystemMessage(content=EXECUTOR_SYSTEM),
                        HumanMessage(content=(
                            f"=== SCHEMAS ===\n{relevant_schema}\n\n"
                            f"=== QUERY PLAN ===\n{json.dumps(plan, indent=2)}\n\n"
                            f"=== CONVERSATION HISTORY ===\n"
                            f"{' '.join(st.session_state.conversation_history)}\n\n"
                            f"=== USER QUESTION ===\n{user_input}\n\n"
                            "Write and run the SQL now using the run_query tool."
                        )),
                    ]

                    for round_num in range(MAX_TOOL_ROUNDS):
                        st.write(f"🔄 Round {round_num + 1}...")
                        response = st.session_state.llm_with_tools.invoke(executor_messages)
                        executor_messages.append(response)

                        done, tool_messages = agents.handle_response(response)

                        if tool_messages:
                            for msg in tool_messages:
                                content = msg.content

                                # Capture the SQL from structured tool calls
                                if hasattr(response, "tool_calls") and response.tool_calls:
                                    for call in response.tool_calls:
                                        if call["name"] == "run_query":
                                            executed_sql = call["args"].get("sql", "")

                                # Capture query results (not schema lookups)
                                if not content.startswith("ERROR") and "columns" not in content[:50]:
                                    query_result = content

                        if done:
                            final_response = response.content

                            # Show the SQL that was executed
                            if executed_sql:
                                with st.expander("🔍 SQL Executed", expanded=False):
                                    st.code(executed_sql, language="sql")

                            st.write("✅ Done!")
                            status.update(label="✅ Query completed", state="complete", expanded=False)
                            break

                        executor_messages.extend(tool_messages)
                    else:
                        final_response = "⚠️ Too many steps — please rephrase your question."
                        status.update(label="⚠️ Incomplete", state="error")

            # Guard against empty response
            if not final_response:
                final_response = "⚠️ Something went wrong — please try again."

            resp_holder.markdown(final_response)

            # Save message with query result and SQL for history rendering
            message_data = {
                "role": "assistant",
                "content": final_response,
                "user_question": user_input,
            }
            if query_result:
                message_data["query_result"] = query_result
            if executed_sql:
                message_data["executed_sql"] = executed_sql
            st.session_state.messages.append(message_data)

            # Add to conversation history (truncated to save tokens)
            response_summary = final_response[:200] + "..." if len(final_response) > 200 else final_response
            st.session_state.conversation_history.append(f"Assistant: {response_summary}")
            if len(st.session_state.conversation_history) > 10:
                st.session_state.conversation_history = st.session_state.conversation_history[-10:]

            # Auto-render chart if plan requested visualization
            wants_graph = (
                plan.get("explicit_visualization", False)
                or plan.get("visualization_recommended", False)
            )

            if wants_graph and query_result:
                with st.spinner("📊 Generating visualization..."):
                    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
                    graph_path = f"{CHART_DIR}/chart_{timestamp}.png"
                    graph_result = generate_graph(query_result, user_input)

                    if graph_result["success"]:
                        exec_result = execute_graph_code(graph_result["code"], graph_path)
                        if exec_result["success"]:
                            st.session_state.messages[-1]["graph_path"] = graph_path
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.image(graph_path, use_container_width=True)
                            with col2:
                                with open(graph_path, "rb") as f:
                                    st.download_button(
                                        "📥 Download", f,
                                        file_name="chart.png", mime="image/png"
                                    )
                        else:
                            st.error(f"❌ Chart failed: {exec_result['error']}")
                    else:
                        st.error(f"❌ {graph_result['error']}")

        break  # exit clarification loop

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🤖 **Powered by:** LangChain + Groq")
with col2:
    st.markdown("💾 **Database:** DuckDB")
with col3:
    st.markdown("📊 **Visualizations:** Matplotlib + Seaborn")