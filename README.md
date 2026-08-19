Trigger api : python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

Trigger Cloudfare : cloudflared tunnel --url http://localhost:8000

Paste the url from the console to N8N http request node.

