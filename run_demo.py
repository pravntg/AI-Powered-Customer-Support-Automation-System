import os
import sys
import sqlite3
import argparse
from langchain_core.messages import HumanMessage

from rag_pipeline import RAGPipeline
from customer_support_graph import create_support_graph
from langgraph.checkpoint.sqlite import SqliteSaver

def run_query(graph, thread_id, query, rag_pipeline, model_name, non_interactive=False):
    print("=" * 80)
    print(f"CUSTOMER: \"{query}\"")
    print("=" * 80)
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "rag_pipeline": rag_pipeline,
            "model_name": model_name
        }
    }
    
    inputs = {
        "current_query": query,
        "messages": [HumanMessage(content=query)],
        "approved": None,
        "supervisor_feedback": ""
    }
    
    # Run the graph (will pause at human_approval node if approval_required is True)
    events = graph.stream(inputs, config, stream_mode="values")
    for event in events:
        pass
        
    state = graph.get_state(config)
    
    # Handle supervisor approval loop
    while "human_approval" in state.next:
        print("\n" + "!" * 80)
        print("!!! DANGER / HIGH-RISK ACTION DETECTED !!!")
        print("Supervisor Approval Required for action in category:", state.values.get("category"))
        print(f"Agent's Draft Response:\n{state.values.get('agent_draft_response')}")
        print("!" * 80 + "\n")
        
        if non_interactive:
            print("[Non-Interactive Mode] Auto-approving the response...")
            approved = True
            feedback = ""
        else:
            choice = input("Do you approve this response? (y/n/exit): ").strip().lower()
            if choice in ['y', 'yes']:
                approved = True
                feedback = ""
            elif choice in ['n', 'no']:
                approved = False
                feedback = input("Provide correction feedback to the agent: ").strip()
            elif choice == 'exit':
                print("Exiting...")
                sys.exit(0)
            else:
                print("Invalid input. Defaulting to rejection.")
                approved = False
                feedback = "Please review your response."
                
        # Update state with supervisor action
        graph.update_state(
            config,
            {"approved": approved, "supervisor_feedback": feedback},
            as_node="human_approval"
        )
        
        # Resume execution
        print(f"\nResuming graph execution (Approved: {approved})...\n")
        events = graph.stream(None, config, stream_mode="values")
        for event in events:
            pass
            
        state = graph.get_state(config)
        
    # Print the validated final response
    final_response = state.values.get("final_response")
    print("-" * 80)
    print(f"FINAL SUPPORT RESPONSE:\n{final_response}")
    print("-" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run the ABC Technologies Customer Support Automation System demo.")
    parser.add_argument("--model", type=str, default="qwen2.5:0.5b", help="Ollama model name (e.g. qwen2.5:0.5b or llama3.1)")
    parser.add_argument("--non-interactive", action="store_true", help="Auto-approve high risk queries without prompting")
    args = parser.parse_args()
    
    print("\n" + "#" * 80)
    print("ABC Technologies AI-Powered Customer Support Automation System")
    print(f"Running with Ollama Model: {args.model}")
    print("#" * 80 + "\n")
    
    # 1. Initialize SQLite connection for persistence
    db_file = "memory.db"
    # Ensure database is clean for the demo
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass
            
    conn = sqlite3.connect(db_file, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # 2. Initialize RAG pipeline
    rag_pipeline = RAGPipeline(doc_dir="documents", model=args.model)
    
    # 3. Create Support Graph
    graph = create_support_graph(checkpointer)
    
    # 4. Run Demonstration Queries
    queries = [
        "What are the pricing plans available for your software?",
        "I forgot my account password.",
        "My application crashes whenever I upload a file.",
        "I need a refund for my annual subscription.",
        "What was my previous support issue?"
    ]
    
    thread_id = "demo_session_user_007"
    
    for i, query in enumerate(queries):
        print(f"\n--- Running Demo Query {i+1} of {len(queries)} ---")
        run_query(graph, thread_id, query, rag_pipeline, args.model, args.non_interactive)
        
    # Clean up DB connection
    conn.close()
    print("Demo execution finished. Database connection closed.")

if __name__ == "__main__":
    main()
