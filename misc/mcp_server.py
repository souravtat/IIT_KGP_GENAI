from mcp.server.fastmcp import FastMCP
#from mcp.server import MCPServer as FastMCP

mcp = FastMCP("Simple Calculator")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()