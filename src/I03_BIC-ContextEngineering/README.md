# Context Engineering — Hands-on Session

Hands-on notebook for the 60-minute Context Engineering slide session.
Builds a Walmart order support assistant one context component at a time.

## Setup

### 1. Install dependencies

```bash
pip install -i https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/pythonhosted-pypi-release-remote/simple/ \
    -r requirements.txt
```

### 2. Place required files

| File | Where | Purpose |
|------|-------|---------|
| `ca-bundle.crt` | repo root | Walmart SSL certificate for the Element Gateway |
| `knowledge_base.md` | repo root | Policy content for RAG (already present) |

### 3. Start the tools API

```bash
uvicorn tools_api.main:app --reload --port 8000
```

Leave this running in a separate terminal for the duration of the session.

Verify: http://localhost:8000/health

### 4. Set your Element API key

Option A — environment variable (recommended):
```bash
export ELEMENT_API_KEY="your-key-here"
```

Option B — the config cell will prompt with `getpass` when the variable is not set.

### 5. Run the notebook

```bash
jupyter notebook context_engineering_handson.ipynb
```

Run cells top to bottom. Do not skip sections — later cells depend on variables set earlier.

## Gateway config

All model and embedding traffic goes through the Walmart Element Gateway.
Edit the config cell (Section 0) to change:
- `GATEWAY_URL`
- `CHAT_MODEL`
- `EMBEDDING_MODEL`
- `API_VERSION`

## State reset

The `refund_order` tool mutates in-memory state in `tools_api/data.py`.
Restart the uvicorn server to reset state before re-running the refund demo.
