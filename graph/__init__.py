"""
graph package — LangGraph AI Data Analyst workflow.

Public exports:
  compiled_graph  — the compiled LangGraph StateGraph (with MemorySaver)
  run_analysis    — async helper to invoke the workflow end-to-end
"""

from graph.workflow import compiled_graph, run_analysis

__all__ = ["compiled_graph", "run_analysis"]
