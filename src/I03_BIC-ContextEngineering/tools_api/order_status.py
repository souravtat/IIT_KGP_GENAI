"""GET order status by order_id."""

from fastapi import APIRouter
from pydantic import BaseModel
from .data import ORDERS

router = APIRouter()


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Get the status, delivery dates, and item list for a specific Walmart order. "
            "Use this when the customer provides an order ID and wants to know its status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The Walmart order ID to look up.",
                }
            },
            "required": ["order_id"],
        },
    },
}


class OrderStatusRequest(BaseModel):
    order_id: str


@router.post("/get_order_status")
def get_order_status(req: OrderStatusRequest):
    order = ORDERS.get(req.order_id)
    if not order:
        return {"error": f"Order {req.order_id!r} not found.", "order_id": req.order_id}
    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "order_date": order["order_date"],
        "delivery_date": order["delivery_date"],
        "payment_method": order["payment_method"],
        "items": order["items"],
        "refunds": order["refunds"],
    }
