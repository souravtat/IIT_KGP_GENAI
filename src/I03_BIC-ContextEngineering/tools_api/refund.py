"""Process a refund for a specific item in an order."""

import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from .data import ORDERS

router = APIRouter()


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "refund_order",
        "description": (
            "Process a refund for a specific item in a Walmart order. "
            "Mutates the order state on the server. "
            "Use only after confirming the order ID and item ID from get_order_status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID containing the item to refund.",
                },
                "item_id": {
                    "type": "string",
                    "description": "The item ID to refund (from the order's items list).",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the refund, e.g. 'damaged on arrival'.",
                },
            },
            "required": ["order_id", "item_id", "reason"],
        },
    },
}


class RefundRequest(BaseModel):
    order_id: str
    item_id: str
    reason: str


@router.post("/refund_order")
def refund_order(req: RefundRequest):
    order = ORDERS.get(req.order_id)
    if not order:
        return {"error": f"Order {req.order_id!r} not found.", "success": False}

    item = next((i for i in order["items"] if i["id"] == req.item_id), None)
    if not item:
        return {
            "error": f"Item {req.item_id!r} not found in order {req.order_id!r}.",
            "success": False,
        }

    refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    refund_amount = round(item["price"] * item["quantity"], 2)

    # Mutate in-memory state — resets on server restart
    order["refunds"].append({
        "refund_id": refund_id,
        "item_id": req.item_id,
        "item_name": item["name"],
        "amount": refund_amount,
        "reason": req.reason,
        "status": "approved",
    })

    return {
        "success": True,
        "refund_id": refund_id,
        "order_id": req.order_id,
        "item_name": item["name"],
        "refund_amount": refund_amount,
        "status": "approved",
        "note": "Refund approved. Funds will appear on the original payment method in 3-5 business days.",
    }
