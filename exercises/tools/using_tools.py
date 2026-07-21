from pprint import pprint

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

from utils import get_attendance_id
from data.customers import (
    get_customer_list as load_customer_list,
    get_customer_annual_spend as load_customer_annual_spend,
    get_customer_office_locations as load_customer_office_locations,
)


model = ChatBedrockConverse(
    model='eu.amazon.nova-pro-v1:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

# Exercise - tool calling
#
#   In this exercise you will provide multiple tools to an LLM agent.
#   The user will ask the agent to answer a question using data from multiple sources,
#       and relying on implicit knowledge.
#
#   In this exercise, the tools use hard-coded local data.
#   In reality, you would likely fetch this data from live databases.
#   The agent's workflow would be the same.
#
#   * Run the code as it is, to see the kinds of information returned by each tool,
#       and see what the agent will return.
#
#   Currently, customer office locations are not available to the agent.
#
#   * Turn `get_customer_office_locations` into a LangChain tool using `@tool`.
#   * Implement it using `load_customer_office_locations`.
#   * Add the new tool to the agent's `tools` list.
#
#   * Try asking questions about customers' offices, for example:
#       - "Which customers have offices in Europe?"
#       - "Which customers have offices in the US and spend more than 2 million a year?"
#

# This is the query we want our agent to be able to answer.
USER_QUERY = "How many high-value customers (spending more than 1 million a year) are there?"

# Display some sample output.
pprint('Sample output from first 5 entries from get_customer_list:')
pprint(load_customer_list()[:5])
print('')
pprint('Sample output from get_customer_annual_spend for customer 1:')
pprint(load_customer_annual_spend(1))
print('')
pprint('Sample output from get_customer_office_locations for customer 1:')
pprint(load_customer_office_locations(1))
print('')


@tool
def get_customer_list() -> list[dict]:
    """Get the list of customers."""
    return load_customer_list()


@tool
def get_customer_annual_spend(id: int) -> int:
    """Get the annual spend made by a given customer."""
    return load_customer_annual_spend(id)


# TODO: Turn this function into a LangChain tool and implement it.
def get_customer_office_locations(id: int) -> list[str]:
    """Get the list of country codes where a given customer has an office."""
    return []


agent = create_agent(
    model=model,
    tools=[
        get_customer_list,
        get_customer_annual_spend,
        # TODO: Add the get_customer_office_locations tool here.
    ],
)

response = agent.invoke({
    'messages': [
        SystemMessage('You can use tools to retrieve information about our customers.'),
        HumanMessage(USER_QUERY),
    ]
})

for message in response['messages']:
    message.pretty_print()
