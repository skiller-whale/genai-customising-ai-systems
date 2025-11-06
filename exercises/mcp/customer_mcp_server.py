from mcp.server.fastmcp import FastMCP
from data.customers import (
    get_customer_list,
    get_customer_annual_spend,
    get_customer_office_locations
)

# Exercise - MCP Servers
#
#  In this exercise you will provide multiple tools to the LLM via an MCP server.
#  The tools are the same as the last exercise, except you will use an MCP server.
#
#  Currently the MCP server implements only one tool - `tool_get_customer_office_locations`.
#  The two other functions are imported above, but not exposed as tools.
#
#  * Define `tool_get_customer_annual_spend(idx: int) -> int` to return the annual spend for a given customer index.
#  * Define `tool_get_customer_list() -> list[dict]` to return the list of customers.`
#  * NOTE: Make sure both tools have a description (this is a Bedrock requirement).
#
#  * Once done, open Continue and make sure the MCP server starts correctly.
#  * Then, ask the LLM in Agent mode questions about customers, e.g.:
#      - How many high-value customers (spending more than 1 million a year) are there?
#      - Which customers have offices in Europe?
#      - Which customers have offices in the US and spend more than 2 million a year?
#      - Which countries do high-value customers (spending more than 1 million a year) have offices in?
#
mcp = FastMCP("Customer MCP")

@mcp.tool(
    description="Get the list of country codes where a given customer has an office."
)
def tool_get_customer_office_locations(idx: int) -> list[str]:
    return get_customer_office_locations(idx)

if __name__ == '__main__':
    mcp.run(transport="stdio")
