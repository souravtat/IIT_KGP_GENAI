"""
FastAPI application for the Context Engineering tools demo.

Each tool lives in its own module and exports a TOOL_SCHEMA dict.
Adding a tool: create a new module, add its router here, include its TOOL_SCHEMA.
"""

from fastapi import FastAPI
from . import order_status, recent_orders, store_inventory, refund

app = FastAPI(title="Walmart Support Tools API")

# Mount each tool's router under /tools/
app.include_router(order_status.router, prefix="/tools")
app.include_router(recent_orders.router, prefix="/tools")
app.include_router(store_inventory.router, prefix="/tools")
app.include_router(refund.router, prefix="/tools")

# Aggregate all tool schemas — the notebook fetches this instead of duplicating definitions
ALL_TOOL_SCHEMAS = [
    order_status.TOOL_SCHEMA,
    recent_orders.TOOL_SCHEMA,
    store_inventory.TOOL_SCHEMA,
    refund.TOOL_SCHEMA,
]


@app.get("/health")
def health():
    return {"status": "ok", "tools": [t["function"]["name"] for t in ALL_TOOL_SCHEMAS]}


@app.get("/tools/schema")
def get_tool_schemas():
    """Return all tool definitions in OpenAI function-calling format."""
    return ALL_TOOL_SCHEMAS
