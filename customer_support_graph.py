import os
import re
import sqlite3
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import ollama

from rag_pipeline import RAGPipeline

# 1. State Structure
class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    customer_info: Dict[str, Any]
    current_query: str
    category: str              # 'Sales', 'Technical', 'Billing', 'Account', 'Memory'
    context: str               # RAG context retrieved
    agent_draft_response: str
    approval_required: bool
    approved: bool             # True, False, or None
    supervisor_feedback: str   # Feedback if rejected by supervisor
    final_response: str

# 2. LLM Call Helper
def call_llm(system_prompt: str, user_prompt: str, model_name: str) -> str:
    try:
        response = ollama.generate(
            model=model_name,
            system=system_prompt,
            prompt=user_prompt,
            options={"temperature": 0.0}  # Low temperature for factual support answers
        )
        return response["response"].strip()
    except Exception as e:
        print(f"Error calling LLM ({model_name}): {e}")
        return "I apologize, but I am experiencing technical difficulties. Please try again later."

# Check if a query is high-risk and requires human approval
def is_high_risk_query(query: str, category: str) -> bool:
    q_lower = query.lower()
    
    # Check for direct keywords representing the 5 restricted actions
    # 1. Refund requests
    # 2. Subscription cancellation
    # 3. Account closure requests
    # 4. Compensation requests
    # 5. Escalation to management
    high_risk_indicators = [
        "refund", "money back", "reimburse",
        "cancel subscription", "stop billing", "unsubscribe",
        "close my account", "delete my account", "close account", "terminate account",
        "compensation", "credit", "discount", "free month",
        "escalate", "speak to manager", "supervisor", "upper management", "complain"
    ]
    
    # If the user is asking for these actions in Billing or Account, it's high risk
    if category in ["Billing", "Account", "Technical"]:
        for indicator in high_risk_indicators:
            if indicator in q_lower:
                return True
    return False

# 3. Node Implementations
def classify_intent_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    query = state["current_query"]
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    q_lower = query.lower()
    
    # 1. Rule-based checks for highest accuracy and speed
    if any(kw in q_lower for kw in ["previous", "earlier", "past", "history", "what did i ask", "last support", "do you remember"]):
        print(f"\n[Node: Classify Intent] classified query: '{query}' -> Category: Memory (Rule-based)")
        return {"category": "Memory"}
        
    if "password" in q_lower or "profile" in q_lower or "username" in q_lower or "account activation" in q_lower or "deactivate" in q_lower:
        print(f"\n[Node: Classify Intent] classified query: '{query}' -> Category: Account (Rule-based)")
        return {"category": "Account"}
        
    if "crash" in q_lower or "error" in q_lower or "upload a file" in q_lower or "fail to upload" in q_lower:
        print(f"\n[Node: Classify Intent] classified query: '{query}' -> Category: Technical (Rule-based)")
        return {"category": "Technical"}

    if "refund" in q_lower or "invoice" in q_lower or "payment" in q_lower or "billing" in q_lower:
        print(f"\n[Node: Classify Intent] classified query: '{query}' -> Category: Billing (Rule-based)")
        return {"category": "Billing"}
        
    if "pricing" in q_lower or "plans" in q_lower or "subscription" in q_lower or "cost" in q_lower:
        print(f"\n[Node: Classify Intent] classified query: '{query}' -> Category: Sales (Rule-based)")
        return {"category": "Sales"}

    # 2. LLM Classifier Fallback
    system_prompt = (
        "You are an expert customer support routing classifier for ABC Technologies.\n"
        "Categorize the user's query into exactly one of these categories:\n"
        "- Sales: Product information, subscription plans, pricing details.\n"
        "- Technical: Application errors, installation issues, login problems, configuration issues.\n"
        "- Billing: Invoice requests, payment issues, refund requests.\n"
        "- Account: Password reset, profile updates, account activation/deactivation.\n"
        "- Memory: Asking about past interactions, previous conversation history, or what was discussed earlier "
        "(e.g., 'What was my previous issue?', 'What did we talk about?', 'Do you remember my name?').\n\n"
        "Return ONLY the category name. Do not explain your choice."
    )
    
    category = call_llm(system_prompt, query, model_name)
    
    # Sanitize category output
    valid_categories = ["Sales", "Technical", "Billing", "Account", "Memory"]
    matched = "Technical" # Default fallback
    for cat in valid_categories:
        if cat.lower() in category.lower():
            matched = cat
            break
            
    print(f"\n[Node: Classify Intent] classified query: '{query}' -> Category: {matched} (LLM-based)")
    return {"category": matched}

def retrieve_rag_context(query: str, category: str, pipeline: RAGPipeline) -> str:
    # Filter documents based on intent category to improve RAG accuracy
    results = pipeline.retrieve(query, top_k=2)
    
    # Format retrieved passages
    context_strs = []
    for r in results:
        context_strs.append(f"[Source: {r['source']}]\n{r['text']}")
    return "\n\n".join(context_strs)

def sales_agent_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    query = state["current_query"]
    pipeline = config["configurable"]["rag_pipeline"]
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    
    context = retrieve_rag_context(query, "Sales", pipeline)
    print(f"[Node: Sales Agent] Retrieved context from: {', '.join(set(re.findall(r'Source: (.*?)]', context)))}")
    
    system_prompt = (
        "You are a Sales Support specialist at ABC Technologies.\n"
        "Answer customer questions about product information, subscription plans, and pricing details.\n"
        "Use the provided context to form your answer. Be professional and helpful.\n\n"
        f"Context:\n{context}"
    )
    
    user_prompt = query
    if state.get("supervisor_feedback"):
        user_prompt += f"\n\n[Correction Feedback from Supervisor]: {state['supervisor_feedback']}"
        
    response = call_llm(system_prompt, user_prompt, model_name)
    approval_required = is_high_risk_query(query, "Sales")
    
    return {
        "agent_draft_response": response,
        "context": context,
        "approval_required": approval_required
    }

def technical_agent_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    query = state["current_query"]
    pipeline = config["configurable"]["rag_pipeline"]
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    
    context = retrieve_rag_context(query, "Technical", pipeline)
    print(f"[Node: Technical Agent] Retrieved context from: {', '.join(set(re.findall(r'Source: (.*?)]', context)))}")
    
    system_prompt = (
        "You are a Technical Support engineer at ABC Technologies.\n"
        "Help customers solve application errors, installation issues, login problems, and configuration issues.\n"
        "Use the technical manual context to troubleshoot. Explain solutions step-by-step.\n\n"
        f"Context:\n{context}"
    )
    
    user_prompt = query
    if state.get("supervisor_feedback"):
        user_prompt += f"\n\n[Correction Feedback from Supervisor]: {state['supervisor_feedback']}"
        
    response = call_llm(system_prompt, user_prompt, model_name)
    approval_required = is_high_risk_query(query, "Technical")
    
    return {
        "agent_draft_response": response,
        "context": context,
        "approval_required": approval_required
    }

def billing_agent_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    query = state["current_query"]
    pipeline = config["configurable"]["rag_pipeline"]
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    
    context = retrieve_rag_context(query, "Billing", pipeline)
    print(f"[Node: Billing Agent] Retrieved context from: {', '.join(set(re.findall(r'Source: (.*?)]', context)))}")
    
    system_prompt = (
        "You are a Billing Support specialist at ABC Technologies.\n"
        "Handle billing issues, invoice requests, and refund requests.\n"
        "Draft responses matching company policies. Remember refund requests require supervisor approval.\n\n"
        f"Context:\n{context}"
    )
    
    user_prompt = query
    if state.get("supervisor_feedback"):
        user_prompt += f"\n\n[Correction Feedback from Supervisor]: {state['supervisor_feedback']}"
        
    response = call_llm(system_prompt, user_prompt, model_name)
    approval_required = is_high_risk_query(query, "Billing")
    
    return {
        "agent_draft_response": response,
        "context": context,
        "approval_required": approval_required
    }

def account_agent_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    query = state["current_query"]
    pipeline = config["configurable"]["rag_pipeline"]
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    
    context = retrieve_rag_context(query, "Account", pipeline)
    print(f"[Node: Account Agent] Retrieved context from: {', '.join(set(re.findall(r'Source: (.*?)]', context)))}")
    
    system_prompt = (
        "You are an Account Support executive at ABC Technologies.\n"
        "Handle queries about password resets, profile updates, and account activation/deactivation.\n"
        "Cite context where relevant. Note that account closures require supervisor approval.\n\n"
        f"Context:\n{context}"
    )
    
    user_prompt = query
    if state.get("supervisor_feedback"):
        user_prompt += f"\n\n[Correction Feedback from Supervisor]: {state['supervisor_feedback']}"
        
    response = call_llm(system_prompt, user_prompt, model_name)
    approval_required = is_high_risk_query(query, "Account")
    
    return {
        "agent_draft_response": response,
        "context": context,
        "approval_required": approval_required
    }

def memory_recall_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    # Extract past conversation context from state messages (which include history loaded by SQLite checkpointer)
    messages = state.get("messages", [])
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    
    # Filter only past Human and AI messages (excluding the current query)
    history = []
    # State messages includes current message, so let's parse all messages up to the second-to-last
    for msg in messages[:-1]:
        role = "Customer" if isinstance(msg, HumanMessage) else "Support Agent"
        history.append(f"{role}: {msg.content}")
        
    history_str = "\n".join(history)
    print(f"[Node: Memory Recall] Retreived {len(messages[:-1])} past messages from SQLite session.")
    
    if not history_str:
        response = "I cannot find any previous support issues or interactions in your active session. How can I help you today?"
    else:
        system_prompt = (
            "You are a Customer Support assistant at ABC Technologies.\n"
            "The customer is asking about their past conversations or previous support issues.\n"
            "Summarize the conversation history provided below to answer their question."
        )
        user_prompt = (
            f"Active Conversation History:\n{history_str}\n\n"
            f"Customer query: {state['current_query']}"
        )
        response = call_llm(system_prompt, user_prompt, model_name)
        
    return {
        "final_response": response,
        "agent_draft_response": response,
        "approval_required": False,
        "messages": [AIMessage(content=response)]
    }

def human_approval_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    # This node executes after the CLI runner inputs human approval status and resumes the graph
    status = "APPROVED" if state.get("approved") else "REJECTED"
    print(f"\n[Node: Human Approval] Supervisor reviewed the request: {status}")
    return {}

def supervisor_evaluation_node(state: SupportState, config: RunnableConfig) -> Dict[str, Any]:
    query = state["current_query"]
    context = state.get("context", "")
    draft = state["agent_draft_response"]
    model_name = config.get("configurable", {}).get("model_name", "qwen2.5:0.5b")
    
    # If it was a memory query, we skip validation against documents
    if state["category"] == "Memory":
        return {
            "final_response": draft,
            "messages": [AIMessage(content=draft)]
        }
        
    system_prompt = (
        "You are the Customer Support Supervisor at ABC Technologies.\n"
        "Review the customer's query, the retrieved company context, and the agent's draft response.\n"
        "Ensure the response is:\n"
        "1. Accurate and fully aligned with the retrieved company policy/technical documents.\n"
        "2. Professional, clear, and does not invent any information.\n\n"
        f"Context:\n{context}\n\n"
        f"Customer Query: {query}\n\n"
        f"Agent Draft Response:\n{draft}"
    )
    
    user_prompt = (
        "Either output the agent's response exactly as-is if it is perfect, or provide "
        "a polished, improved version of the response. Output ONLY the response text to send "
        "to the customer, with no headers, introductions, or explanations."
    )
    
    final_resp = call_llm(system_prompt, user_prompt, model_name)
    print(f"[Node: Supervisor Evaluation] Polished and validated final response.")
    return {
        "final_response": final_resp,
        "messages": [AIMessage(content=final_resp)]
    }

# 4. Routing Functions
def route_intent(state: SupportState) -> str:
    return state["category"]

def route_after_agent(state: SupportState) -> str:
    if state["approval_required"]:
        return "human_approval"
    return "supervisor_evaluation"

def route_after_approval(state: SupportState) -> str:
    if state.get("approved"):
        return "supervisor_evaluation"
    else:
        # Loop back to agent if supervisor rejects
        category = state["category"]
        if category == "Sales":
            return "sales_agent"
        elif category == "Technical":
            return "technical_agent"
        elif category == "Billing":
            return "billing_agent"
        elif category == "Account":
            return "account_agent"
        return END

# 5. Graph Compilation
def create_support_graph(checkpointer: SqliteSaver) -> StateGraph:
    workflow = StateGraph(SupportState)
    
    # Add Nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("sales_agent", sales_agent_node)
    workflow.add_node("technical_agent", technical_agent_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("account_agent", account_agent_node)
    workflow.add_node("memory_recall", memory_recall_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("supervisor_evaluation", supervisor_evaluation_node)
    
    # Add Edges
    workflow.add_edge(START, "classify_intent")
    
    # Intent Routing
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "Sales": "sales_agent",
            "Technical": "technical_agent",
            "Billing": "billing_agent",
            "Account": "account_agent",
            "Memory": "memory_recall"
        }
    )
    
    # Agent Routing (to human approval or supervisor evaluation)
    for agent_node in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
        workflow.add_conditional_edges(
            agent_node,
            route_after_agent,
            {
                "human_approval": "human_approval",
                "supervisor_evaluation": "supervisor_evaluation"
            }
        )
        
    # Human Approval Routing (to supervisor or loop back to agent)
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "supervisor_evaluation": "supervisor_evaluation",
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent"
        }
    )
    
    workflow.add_edge("supervisor_evaluation", END)
    workflow.add_edge("memory_recall", END)
    
    # Compile with human_approval as an interrupt node
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"]
    )
    
    return graph
