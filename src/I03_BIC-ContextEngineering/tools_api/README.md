# Tools API

FastAPI service backing the notebook's tool calls.

## Start the server

```bash
cd /path/to/BIC-ContextEngineering
uvicorn tools_api.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check |
| GET | `/tools/schema` | All tool definitions in OpenAI format |
| POST | `/tools/get_order_status` | Order details by order_id |
| POST | `/tools/list_recent_orders` | Order list by customer_id |
| POST | `/tools/check_store_inventory` | Stock and aisle by item_name + store_id |
| POST | `/tools/refund_order` | Approve refund; mutates in-memory state |

## State reset

`refund_order` mutates an in-memory dict in `data.py`.
To reset state for a re-run of the refund demo, restart the server.

## Interactive docs

While the server is running: http://localhost:8000/docs
