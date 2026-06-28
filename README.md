# AI-Powered Customer Support Automation System (LangGraph)

This repository contains the complete implementation of **Assignment - 2: AI-Powered Customer Support Automation System** built with **LangGraph**, **Ollama**, and **SQLite**. 

The system accepts customer queries, classifies their intent, retrieves relevant company context via a RAG pipeline, routes queries to specialized support agents, enforces human-in-the-loop review for high-risk requests, utilizes a supervisor agent to audit and polish replies, and remembers the conversation history using a local SQLite-backed memory saver.

---

## 1. System Architecture

The workflow is modeled as a stateful graph in LangGraph:

```
                  +-----------------------+
                  |         START         |
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  |    classify_intent    |
                  +-----------+-----------+
                              |
       +----------------------+----------------------+--------------------+
       | (Sales)              | (Technical)          | (Billing)          | (Account)
       v                      v                      v                    v
+------+------+        +------+------+        +------+------+      +------+------+
| sales_agent |        | technical_  |        | billing_    |      | account_    |
|   (RAG)     |        | agent (RAG) |        | agent (RAG) |      | agent (RAG) |
+------+------+        +------+------+        +------+------+      +------+------+
       |                      |                      |                    |
       +----------------------+----------------------+--------------------+
                              |
                              v
                +-------------+-------------+
                |   is_high_risk_query?     |
                +-------------+-------------+
                              |
               +--------------+--------------+
               | (Yes)                       | (No)
               v                             |
     ==================                      |
     = HUMAN APPROVAL =  <-----+             |
     =   (Interrupt)  =        | (No)        |
     ==================        |             |
               |               |             |
               +---------------+             |
               | (Yes)                       |
               v                             v
     +---------+---------+         +---------+---------+
     |   supervisor_     |         |   supervisor_     |
     |   evaluation      |         |   evaluation      |
     +---------+---------+         +---------+---------+
               |                             |
               +--------------+--------------+
                              |
                              v
                           [ END ]
```

### Components:
1. **Intent Classifier Node**: Routes queries dynamically using LLM classification to *Sales*, *Technical*, *Billing*, *Account*, or *Memory*.
2. **RAG Pipeline (`rag_pipeline.py`)**: Utilizes custom chunking of company documents (`company_policy.txt`, `pricing_guide.txt`, `technical_manual.txt`, `faq_document.txt`).
   - *Dual-Mode Retrieval*: Performs cosine similarity search using Ollama embeddings (`qwen2.5:0.5b` or `llama3.1`). If the Ollama model is busy loading, it instantly falls back to an in-memory TF-IDF index using `scikit-learn` to prevent service delay.
3. **Specialized Agents**:
   - **Sales Agent**: Answers plans and pricing queries using the Pricing Guide and FAQ.
   - **Technical Agent**: Troubleshoots application errors and file crashes using the Technical Manual and FAQ.
   - **Billing Agent**: Handles invoices and billing policies using the Policy Document and FAQ.
   - **Account Agent**: Performs account resets and activations using the Technical Manual and FAQ.
4. **Human-in-the-Loop Interruption Node**: Automatically triggers if a high-risk request is made (refunds, cancellations, closures, compensation, escalations). The graph pauses execution and waits for a supervisor's approval/rejection. Rejections route back to the agent with feedback for regeneration.
5. **Supervisor Agent Node**: Validates the agent's drafted answer against the retrieved context to guarantee absolute compliance, politeness, and facts, rewriting it if improvements are needed.
6. **SQLite Memory Saver**: Compiles with `SqliteSaver` to automatically write the message threads to `memory.db` for stateful persistence.
7. **Memory Recall Agent**: Directly answers memory queries (e.g., *"What was my previous support issue?"*) by pulling history from SQLite and summarizing the conversation turns.

---

## 2. Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Ollama installed and running locally with the following models:
  - `qwen2.5:0.5b` (Recommended for fast CPU execution, 380 MB)
  - `llama3.1` (Primary 8B model, 4.9 GB)

### Install Dependencies
Run the following command to install the required Python libraries:
```bash
pip install langgraph langchain-core langchain-ollama numpy scikit-learn pypdf langgraph-checkpoint-sqlite
```

---

## 3. How to Run the Demonstration

The script `run_demo.py` executes the 5 required demonstration queries in sequence under the same thread session `demo_session_user_007`.

### Mode A: Interactive Supervisor Approval (Recommended)
This runs the demo queries, pausing at the refund query to ask for your manual input (`y` for approve, `n` for reject) in the terminal:
```bash
# Using Qwen 2.5 (Fast CPU execution, downloads in seconds)
python run_demo.py --model qwen2.5:0.5b

# Using Llama 3.1 (Slow initial load on CPU, high quality)
python run_demo.py --model llama3.1
```

### Mode B: Automated Non-Interactive Mode
This runs the full suite and automatically signs off/approves the refund query without prompting:
```bash
python run_demo.py --model qwen2.5:0.5b --non-interactive
```

---

## 4. Verification Files
- **Database**: `memory.db` is created in the root folder upon execution. It contains the session messages.
- **Documents**: `documents/` contains the policy, pricing, technical, and FAQ text files.
