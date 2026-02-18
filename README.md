<<<<<<< HEAD
# Data Talker

## Architecture
This project is structured in a modular way, promoting separation of concerns.

## File Descriptions
- `main.py`: The entry point of the application.
- `data_processor.py`: Contains the logic to process the data.
- `api.py`: Handles API interactions.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/aryanparab/Data-Talker.git
   cd Data-Talker
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Features
- Process various data formats.
- Integrate with third-party APIs.
- User-friendly command-line interface.

## Examples
- Example command to run the application:
  ```bash
  python main.py --input data.json
  ```
=======
# 🤖 Conversational Database Agent

A natural language interface for querying DuckDB databases using AI-powered agents. Ask questions about your sales data in plain English and get answers with optional visualizations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.2.8-green.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-1.4.4-yellow.svg)
![Groq](https://img.shields.io/badge/Groq-Llama--3.3-orange)

## ✨ Features

- **🗣️ Natural Language Queries** - Ask questions about your data in plain English
- **🧠 Multi-Model Architecture** - Specialized LLMs for planning, SQL generation, and execution
- **📊 Data Visualization** - Automatic chart generation (bar, line, scatter, pie, heatmap)
- **🔍 Schema Exploration** - Automatically discovers and describes database tables
- **💡 Smart Clarification** - Asks clarifying questions when queries are ambiguous
- **🔒 Safe & Secure** - Only SELECT queries are allowed; your data never leaves the system

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for:
  - **Groq** (required for tool calling) - [Get free API key](https://console.groq.com/)
  - **Hugging Face** (optional, for specialized SQL model) - [Get free API key](https://huggingface.co/settings/tokens)

### Installation

1. **Clone and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_...
HUGGINGFACE_API_KEY=hf_...
DB_PATH=salesdata.db
```

3. **Seed the database (if needed):**
```bash
python db.py
```

4. **Start the agent:**
```bash
python chatter.py
```

### Usage Example

```
============================================================
  Multi-Model DB Agent
  - Planner : Qwen/Qwen2.5-72B-Instruct (HF API)
  - SQL     : Qwen/Qwen2.5-Coder-7B-Instruct (HF API)
  - Executor: Groq  (tool calling)
  - Viz     : Groq  (on-demand)
============================================================

  Tables available: salesperson, orders, training, bonus_pay, agent_commissions

Type 'quit' to exit.
------------------------------------------------------------
You: Show me total sales by salesperson

  [planner] Analyzing query...
  [planner] Tables: ['orders', 'salesperson']
  [planner] Logic : Join orders with salesperson, group by name, sum amount

  [executor] Running with Groq tool-calling...

Agent: Here are the total sales by salesperson:

| Salesperson | Total Sales |
|-------------|-------------|
| Tom         | $2,200      |
| Josh        | $2,700      |
| Bob         | $2,640      |
| Alice       | $1,600      |
| Joe         | $700        |

Query Completed!
```

## 📁 Project Structure

```
botman/
├── agents.py           # Core agent logic (planning, SQL generation)
├── agent_tools.py      # Database tools (run_query, get_schema, list_tables)
├── chatter.py          # Main CLI interface
├── graph_agent.py     # Visualization generation
├── db.py              # Database setup and sample data
├── huggingface.py     # Local Hugging Face model wrapper
├── requirements.txt    # Python dependencies
├── salesdata.db       # DuckDB database (created on first run)
└── README.md          # This file
```

## 🏗️ Architecture

### Multi-Model Pipeline

```
User Query → Planner Model → Query Plan → SQL Model → SQL Query
                                                     ↓
                                              Database (DuckDB)
                                                     ↓
                                              Results → Executor → Response
                                                     ↓
                                              Visualization (Optional)
```

### Models Used

| Component | Model | Provider | Purpose |
|-----------|-------|----------|---------|
| **Planner** | Qwen/Qwen2.5-72B-Instruct | Hugging Face (Free) | Analyze queries, create execution plans |
| **SQL Generator** | Qwen/Qwen2.5-Coder-7B-Instruct | Hugging Face (Free) | Generate optimized SQL queries |
| **Executor** | llama-3.3-70b-versatile | Groq | Execute queries with tool calling |
| **Visualization** | llama-3.3-70b-versatile | Groq | Generate chart code |

> **Note:** All components fall back to Groq's llama-3.3-70b-versatile if API keys are unavailable.

### Database Schema

The sample database includes two datasets:

#### DS-1: Sales Data (4 tables)

| Table | Description | Rows |
|-------|-------------|------|
| `salesperson` | Sales agent information | 5 |
| `orders` | Sales orders with amounts | 10 |
| `training` | Training windows (date validity periods) | 8 |
| `bonus_pay` | Bonus tier thresholds by year | 4 |

#### DS-2: Agent Commissions (1 wide table)

| Table | Description | Columns |
|-------|-------------|---------|
| `agent_commissions` | Monthly commissions for 10 agents | 41 (36 monthly + 5 metadata) |

## 💬 Example Queries

### Basic Queries
```
You: How many orders are there?
You: What's the average order amount?
You: List all salespeople
```

### Aggregations
```
You: Show total sales by salesperson
You: Which month had the highest sales in 2023?
You: What's the average commission per agent?
```

### Joins & Analysis
```
You: Which salespeople had orders outside their training window?
You: Calculate bonus pay for each salesperson in 2013
You: Show agents with their upline managers
```

### Visualizations
```
You: Create a bar chart of sales by salesperson
You: Show monthly commission trends as a line chart
You: Create a pie chart of sales by agency
```

### Schema Exploration
```
You: Describe the orders table
You: What data do you have about agents?
You: Show me the schema for all tables
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for Groq (tool calling) |
| `HUGGINGFACE_API_KEY` | No | API key for Hugging Face (specialized models) |
| `DB_PATH` | No | Path to DuckDB database (default: `salesdata.db`) |

### Model Configuration (in `agents.py`)

```python
# Planner Model - good at instructions and JSON output
PLANNER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# SQL Model - specialized for text-to-SQL
SQL_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Use Hugging Face API (set to False to use only Groq)
USE_HUGGINGFACE_API = True
```

## 🛠️ Development

### Running Tests
```bash
# Test the graph agent visualization
python graph_agent.py
```

### Adding New Tools

1. Define the tool in `agent_tools.py` using `@tool` decorator
2. Register it in `get_tools()` function
3. Update prompts if needed

### Local Model Support

For offline usage, use the local Hugging Face wrapper:
```python
from huggingface import HuggingFaceChatModel

model = HuggingFaceChatModel(model_name="meta-llama/Llama-2-7b-chat-hf")
```

> **Note:** Local models require significant GPU/CPU resources and may not support tool calling natively.

## 📋 Dependencies

Key dependencies include:

- **langchain** - LLM orchestration
- **langchain-groq** - Groq integration
- **duckdb** - Embedded SQL database
- **groq** - Groq API client
- **huggingface_hub** - Hugging Face API access
- **pandas** - Data manipulation
- **matplotlib** & **seaborn** - Visualization

See `requirements.txt` for full list.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Support for more database backends
- Additional visualization types
- Improved error handling
- Web interface (Streamlit/Gradio)
- API server mode

## 📄 License

MIT License - feel free to use and modify for your projects.

---

Built with ❤️ using LangChain, DuckDB, and Groq

>>>>>>> c297763 (new changes)
