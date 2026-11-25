"""
In-memory store for fake customer and order data.
State is module-level and mutable — the refund tool writes here.
Restart the server to reset state (noted in README).
"""

CUSTOMER = {
    "id": "C-1001",
    "name": "Maria Chen",
    "membership_tier": "Walmart+",
    "city": "Dallas, TX",
    "preferred_store": "Store #4482",
}

ORDERS = {
    # Recent, refund-eligible — the running scenario order
    "8842": {
        "order_id": "8842",
        "customer_id": "C-1001",
        "status": "delivered",
        "order_date": "2026-08-20",
        "delivery_date": "2026-08-24",
        "payment_method": "Visa ending 4521",
        "items": [
            {"id": "ITEM-001", "name": "NutriBlend Pro Blender", "price": 49.99, "quantity": 1},
            {"id": "ITEM-002", "name": "Glass Jar Set (4-pack)", "price": 12.99, "quantity": 1},
        ],
        "refunds": [],
    },
    # Outside the 90-day return window
    "7731": {
        "order_id": "7731",
        "customer_id": "C-1001",
        "status": "delivered",
        "order_date": "2026-04-10",
        "delivery_date": "2026-04-15",
        "payment_method": "Visa ending 4521",
        "items": [
            {"id": "ITEM-003", "name": "Instant Pot Duo 6Qt", "price": 79.99, "quantity": 1},
        ],
        "refunds": [],
    },
    # Still in transit
    "9001": {
        "order_id": "9001",
        "customer_id": "C-1001",
        "status": "in_transit",
        "order_date": "2026-08-29",
        "delivery_date": "2026-09-02",
        "payment_method": "Walmart Pay",
        "items": [
            {"id": "ITEM-004", "name": "Cascade Dish Pods (85ct)", "price": 18.99, "quantity": 2},
        ],
        "refunds": [],
    },
}

STORE_INVENTORY = {
    "Store #4482": {
        "NutriBlend Pro Blender": {"in_stock": True, "quantity": 3, "aisle": "A12"},
        "Instant Pot Duo 6Qt":   {"in_stock": False, "quantity": 0, "aisle": "B7"},
        "Glass Jar Set (4-pack)": {"in_stock": True, "quantity": 11, "aisle": "A15"},
        "Cascade Dish Pods (85ct)": {"in_stock": True, "quantity": 24, "aisle": "C3"},
    },
    "Store #1100": {
        "NutriBlend Pro Blender": {"in_stock": True, "quantity": 1, "aisle": "A9"},
        "Instant Pot Duo 6Qt":   {"in_stock": True, "quantity": 2, "aisle": "B4"},
        "Glass Jar Set (4-pack)": {"in_stock": False, "quantity": 0, "aisle": "A18"},
        "Cascade Dish Pods (85ct)": {"in_stock": True, "quantity": 8, "aisle": "C1"},
    },
}
