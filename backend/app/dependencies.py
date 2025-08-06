
# backend/app/dependencies.py
from fastapi import Request
from typing import Any

def get_vector_service(request: Request):
    return request.app.state.vector_service

def get_graph_service(request: Request):
    return request.app.state.graph_service

def get_llm_client(request: Request) -> Any:
    """Get the current LLM client (dynamically chosen based on config)."""
    # Call the function to get current client based on latest config
    get_client_func = request.app.state.get_llm_client
    return get_client_func()