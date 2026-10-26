import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tahoe_operator.config import get_settings


class HouseMCPClient:
    """Small client used by LangGraph nodes; every operational tool call crosses House MCP."""

    def __init__(self, url: str = "") -> None:
        self.url = url or get_settings().mcp_url

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with streamable_http_client(self.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
        if result.isError:
            raise RuntimeError(f"House MCP tool {name} failed: {result.content}")
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"text": text}
        return {}
