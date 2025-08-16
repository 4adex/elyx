from src import OrchestatedAgent

agent = OrchestatedAgent()

graph = agent.graph

mermaid_text = graph.get_graph().draw_mermaid()

print(mermaid_text)