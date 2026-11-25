"""Check store inventory for an item at a specific store."""

from fastapi import APIRouter
from pydantic import BaseModel
from .data import STORE_INVENTORY

router = APIRouter()


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_store_inventory",
        "description": (
            "Check whether an item is in stock at a specific Walmart store, and which aisle it is in. "
            "Use this ONLY when the customer explicitly asks about in-store availability or wants to visit a store."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "Name of the item to check.",
                },
                "store_id": {
                    "type": "string",
                    "description": "The store identifier, e.g. 'Store #4482'.",
                },
            },
            "required": ["item_name", "store_id"],
        },
    },
}


class InventoryRequest(BaseModel):
    item_name: str
    store_id: str


@router.post("/check_store_inventory")
def check_store_inventory(req: InventoryRequest):
    store = STORE_INVENTORY.get(req.store_id)
    if not store:
        return {"error": f"Store {req.store_id!r} not found in inventory system.", "in_stock": False}

    # Case-insensitive item lookup
    item_data = next(
        (v for k, v in store.items() if k.lower() == req.item_name.lower()),
        None,
    )
    if item_data is None:
        return {
            "item_name": req.item_name,
            "store_id": req.store_id,
            "in_stock": False,
            "note": "Item not found in this store's inventory.",
        }

    return {
        "item_name": req.item_name,
        "store_id": req.store_id,
        "in_stock": item_data["in_stock"],
        "quantity": item_data["quantity"],
        "aisle": item_data["aisle"],
    }
