#!/usr/bin/env python3
"""
Enhanced proxy service with comprehensive debugging and correct port routing
"""
import httpx
import uvicorn
import traceback
from fastapi import FastAPI, Request, Response
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ollama-proxy")

app = FastAPI(title="Ollama Proxy for Graphiti Compatibility")

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(request: Request, path: str):
    """
    A more robust proxy that logs details and handles path forwarding.
    This will catch both /v1/responses and /v1/chat/completions.
    """
    # 1. Log the actual incoming request path
    actual_path = request.url.path
    logger.info(f"Incoming request for: {actual_path}")

    # Determine the target path
    target_path = "v1/chat/completions"
    if actual_path == "/v1/responses":
        logger.info(f"🔄 Mapping {actual_path} -> /{target_path}")
    else:
        # If it's not /v1/responses, we just pass it through.
        # This helps debug what Graphiti is *actually* calling.
        target_path = f"v1/{path}"
        logger.info(f"Passing through request for: /{target_path}")

    # 2. Correct Ollama URL - FIXED to use internal container port 11434
    ollama_url = f"http://ollama:11434/{target_path}"
    logger.info(f"Target URL: {ollama_url}")

    try:
        body = await request.body()

        # 3. Forward headers selectively
        headers_to_forward = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length", "accept-encoding"]
        }
        # httpx will set its own Host, Content-Length, etc.

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.request(
                request.method,
                ollama_url,
                content=body,
                headers=headers_to_forward,
                params=dict(request.query_params)
            )

            logger.info(f"✅ Ollama response: {response.status_code}")

            # Exclude certain headers from the response to the client
            response_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in ["content-encoding", "transfer-encoding"]
            }

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers
            )
    except Exception as e:
        # 4. Improved error logging
        error_details = traceback.format_exc()
        logger.error(f"❌ Proxy error processing {actual_path}: {e}\n{error_details}")
        return Response(content=f"Proxy error: {e}\n{error_details}", status_code=502)

if __name__ == "__main__":
    logger.info("🚀 Starting Ollama Proxy for Graphiti Compatibility")
    uvicorn.run(app, host="0.0.0.0", port=11434)