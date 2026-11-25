"""List recent orders for a customer."""

from fastapi import APIRouter
from pydantic import BaseModel
from .data import ORDERS

router = APIRouter()


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_recent_orders",
        "description": (
            "List summary information for all recent orders belonging to a customer. "
            "Use this when the customer does not know their order ID and needs to find it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID whose orders to list.",
                }
            },
            "required": ["customer_id"],
        },
    },
}


class RecentOrdersRequest(BaseModel):
    customer_id: str


@router.post("/list_recent_orders")
def list_recent_orders(req: RecentOrdersRequest):
    customer_orders = [
        {
            "order_id": o["order_id"],
            "status": o["status"],
            "order_date": o["order_date"],
            "delivery_date": o["delivery_date"],
            "item_count": len(o["items"]),
            "items_summary": ", ".join(i["name"] for i in o["items"]),
        }
        for o in ORDERS.values()
        if o["customer_id"] == req.customer_id
    ]
    if not customer_orders:
        return {"error": f"No orders found for customer {req.customer_id!r}.", "orders": []}
    return {"customer_id": req.customer_id, "orders": customer_orders}
