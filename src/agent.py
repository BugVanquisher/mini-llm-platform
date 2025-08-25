# src/agent.py
import re
from typing import Dict, Any
from rag import query_index

# Define available tools
def tool_rag(query: str):
    results = query_index(query, top_k=3)
    docs = [r[0] for r in results]
    return "\n".join(docs)

def tool_calc(expr: str):
    try:
        # eval for demo — in prod, replace with a safe math parser
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "rag": tool_rag,
    "calc": tool_calc,
}


def parse_action(output: str):
    """
    Parse model output in the format:
    Action: tool[input]
    """
    match = re.search(r"Action:\s*(\w+)\[(.+)\]", output)
    if match:
        return match.group(1), match.group(2)
    return None, None


async def run_agent(query: str, client, model_name: str, ollama_url: str) -> Dict[str, Any]:
    """
    Minimal agent loop using ReAct-style prompting.
    """
    system_prompt = """You are an intelligent assistant.
You can take actions using the following tools:
- rag[query] → retrieve context from documents
- calc[expression] → calculate an expression

Format:
Thought: reasoning
Action: tool_name[input]

Once you have enough information:
Final Answer: <your answer>
"""

    history = []
    while True:
        prompt = system_prompt + "\n\n" + "\n".join(history) + f"\nUser: {query}"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        r = await client.post(f"{ollama_url}/api/generate", json=payload, timeout=60.0)
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        history.append(response)

        # Check if model gave Final Answer
        if "Final Answer:" in response:
            final = response.split("Final Answer:")[-1].strip()
            return {"query": query, "trace": history, "answer": final}

        # Otherwise parse Action
        tool, arg = parse_action(response)
        if tool in TOOLS:
            tool_output = TOOLS[tool](arg)
            history.append(f"Observation: {tool_output}")
        else:
            # If no valid tool → break
            return {"query": query, "trace": history, "answer": response}