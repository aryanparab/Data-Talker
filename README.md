# Data Talker

[Live demo](https://data-talker-demo.onrender.com) · [Release source](https://github.com/aryanparab/Data-Talker/tree/prod)

A conversational analytics workspace that turns questions into inspectable SQL, result tables, and charts. Ask a question, refine it in a follow-up, and check exactly how the answer was calculated.

The public demo uses **2,832 entirely synthetic rows across five related tables**: sales representatives, orders, eligibility periods, bonus thresholds, and commissions. No case-study workbook or original database is included in the deployment image.

## What makes this more than a SQL prompt?

- **Live catalog grounding.** The planner receives actual column types, row counts, declared foreign keys, and a separate metric contract. Dataset-specific rules and example queries live in `datatalker/demo.py`, not the system prompt. There are no question-to-answer keyword shortcuts.
- **Clarification before guessing.** Ambiguous agent names are checked against real manager/agency contexts. The app asks you to include the context, rather than silently choosing one.
- **An execution boundary.** SQLGlot validates one SELECT query against allowed tables and functions. DuckDB opens read-only in a separate process with external access and extension loading disabled, a 128 MB engine memory limit, a six-second CPU limit, and a twenty-second wall-clock timeout. Results are capped at 200 rows and about 200 KB.
- **Visible evidence.** Results, scope assumptions, executed SQL, latency, model-call counts, and token usage are shown together. The result summary formats executed values directly instead of asking another model to invent a narrative.
- **Safe chart rendering.** The model proposes only a chart type and result-column names. The server validates them; the browser draws SVG. No model-generated Python or JavaScript is executed.
- **Explicit verified examples.** Four reviewed queries run without model calls. They are clearly labeled and remain useful when the provider is unavailable or the public allowance is reached. Live questions are never silently replaced with canned answers.
- **Follow-up context.** Up to six recent question/SQL pairs are held in browser session storage and supplied to the planner. Uncheck “Continue the previous query” to start an independent question. Grouped follow-ups are checked for accidental collapse into a scalar. Clear session removes that context; reloading clears the displayed conversation.

## Run locally

Requires Python 3.12 (the locally tested and deployed version).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt
cp .env.example .env
# Add your Groq API key to .env
python -m datatalker.demo
uvicorn datatalker.app:app --host 127.0.0.1 --port 8070
```

Open <http://localhost:8070> for the project landing page or <http://localhost:8070/app> for the analytics workspace. Verified examples work without a model key. `GROQ_MODEL` selects a model available to your Groq account; the default is `openai/gpt-oss-120b`.

## Architecture

```text
Browser: question + recent query context
  → FastAPI: input bounds, public allowance, entity ambiguity check
  → Groq: structured query plan grounded in catalog + metric contract
  → SQLGlot: one SELECT, allowed sources and functions
  → isolated worker: read-only DuckDB, bounded execution
  → validated result table + declarative SVG chart + executed SQL
```

The planner can repair a rejected query once. It cannot change the database. The public runtime imports only `datatalker/`; the older Streamlit/LangChain experiments at the repository root are legacy code and are not deployed or dependencies of this app. Use the commands above rather than `streamer.py`.

## Deploy on Render

The root `Dockerfile` builds one service containing both the API and static frontend. `.dockerignore` allows only the new runtime, frontend, and pinned runtime dependencies into the build context. The synthetic database is generated during the build and the server runs as a non-root user.

1. Create a Render Web Service from this repository's `prod` branch.
2. Select **Docker** and **Free**. Leave the root directory blank.
3. Add `GROQ_API_KEY` as a protected environment variable. Optional: `GROQ_MODEL` and `DAILY_MODEL_CALLS`.
4. Set the health-check path to `/api/health`. Deploy.

`render.yaml` provides the equivalent blueprint configuration. No external database, persistent disk, separate frontend host, or original workbook is required. Render supplies `PORT` automatically.

Free services sleep when idle and can have a slow first request. Free instance hours are shared across the workspace; multiple running free services may exhaust that allowance. Groq has its own token/request limits. The app permits 15 live questions per client address per hour and defaults to 100 model calls per UTC day per server process. A query uses one call, or two if repaired. Counters reset on restart and are **not a durable billing limit**. Set provider-side spending controls for stronger cost protection.

## Validation

```sh
python -m unittest discover -s tests -v
# Optional: requires the running app and a configured Groq key; consumes model calls.
python -m tests.live_eval http://localhost:8070
```

The 18-test offline suite checks independently aggregated reference results for all four examples, SQL rejection cases, worker timeout, row limits, chart validation, CTE/window support, and ambiguous names. The opt-in live evaluator compares generated SQL results with reference queries and checks follow-up context. Its report goes to `/tmp/data-talker-live-eval.json`; it is a small regression set, not a general accuracy benchmark.

## Limits and privacy

Only synthetic data is supported by this public deployment. Live questions, recent question/SQL context, and the synthetic catalog are sent to Groq. There is no user-upload endpoint or arbitrary database connector. Query history stays in browser session storage and request bodies are not explicitly logged by this app.

Read-only mode, validation, subprocess limits, and a container are defense in depth, not a complete sandbox for hostile SQL or sensitive tenant isolation. AI-generated SQL can still be semantically wrong; review the visible SQL and metric assumptions. Do not connect private production data without authentication, authorization, stronger workload isolation, and durable quotas.
