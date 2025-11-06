from pprint import pprint
import boto3
from utils import get_attendance_id

from data.customers import get_customer_list, get_customer_annual_spend, get_customer_office_locations

session = boto3.Session(
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

client = session.client(
    service_name='bedrock-runtime', region_name='eu-west-1',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/'
)

# Exercise - tool calling
#
#   In this exercise you will provide multiple tools to the LLM.
#   Then the user will ask the LLM to answer a question using data from multiple sources,
#       and relying on implicit knowledge.
#
#  In this exercise, the tools use hard-coded local data -
#       in reality, you would likely fetch this data from live databases.
#  The LLM's workflow would be the same.
#
#  * Run the code as it is, to see the kinds of information returned by each tool,
#       and see what the LLM will return.
#
#  Currently there are tools that aren't actually called - you will need to implement that.
#
#  * There is a JSON schema defined for the `get_customer_annual_spend` and `get_customer_list` tools.
#     Add a schema for the `get_customer_office_locations` tool (it takes one parameter, `id`, of type `integer`).
#  * Implement the code to call the `get_customer_office_locations` tool when the LLM requests it.
#     You can see examples of how to call the other tools in the code below.
#
#  * Try asking questions about customers' offices, for example:
#     - "Which customers have offices in Europe?"
#     - "Which customers have offices in the US and spend more than 2 million a year?""
#

# This is the query we want our LLM to be able to answers
USER_QUERY = "How many high-value customers (spending more than 1 million a year) are there?"

# Display some sample output
pprint('Sample output from first 5 entries from get_customer_list:')
pprint(get_customer_list()[:5])
print('')
pprint('Sample output from get_customer_annual_spend for customer 1:')
pprint(get_customer_annual_spend(1))
print('')
pprint('Sample output from get_customer_office_locations for customer 1:')
pprint(get_customer_office_locations(1))
print('')

# TODO add a toolSpec for `get_customer_office_locations`
TOOL_CONFIG = { "tools":
    [
        {
            "toolSpec": {
                "name": "get_customer_annual_spend",
                "description": "Get the annual spend made by a given customer.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": { "id": { "type": "integer" } },
                        "required": ["id"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_customer_list",
                "description": "Get the list of customers.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        }
        # TODO: YOUR CODE GOES HERE
    ]
}


sysprompt = 'You can use tools to retrieve information about our customers.'
messages = [
    {'role': 'user', 'content': [{'text': USER_QUERY}]},
]

print(messages)

# Loop until we get a response that is not 'toolUse'
while True:
    result = client.converse(
        modelId='eu.amazon.nova-pro-v1:0',
        system=[{ "text": sysprompt }],
        toolConfig=TOOL_CONFIG,
            messages=messages
    )

    content = result['output']['message']['content']
    messages.append(result['output']['message'])

    pprint(content)

    # Iterate over the parts of the LLM's message
    # For any parts that are tool use requests, we create a matching tool use response
    tool_use_responses = []
    for response_part in content:
        if 'toolUse' not in response_part:
            continue

        tool_use = response_part['toolUse']
        tool_name = tool_use['name']
        tool_input = tool_use['input']
        tool_use_id = tool_use['toolUseId']

        if tool_name == 'get_customer_list':
            customer_list = get_customer_list()
            tool_use_responses.append({
                'toolResult': {
                    'toolUseId': tool_use_id,
                    'content': [{'json': {'customers': customer_list}}]
                }
            })

        elif tool_name == 'get_customer_annual_spend':
            customer_annual_spend = get_customer_annual_spend(tool_input['id'])
            tool_use_responses.append({
                'toolResult': {
                    'toolUseId': tool_use_id,
                    'content': [{'json': {'annual_spend': customer_annual_spend}}]
                }
            })

        elif tool_name == 'get_customer_office_locations':
            # TODO get data using get_customer_office_locations
            customer_office_locations = []
            tool_use_responses.append({
                'toolResult': {
                    'toolUseId': tool_use_id,
                    'content': [{'json': {'office_locations': customer_office_locations}}]
                }
            })

        else:
            # Unknown tool, we should log and manage the error here.
            print(f'Unknown tool: {tool_name}')
            exit(1)

    if len(tool_use_responses) == 0:
        break
    else:
        # Create a message to send back all of the tool use responses we've created.
        messages.append({'role': 'user', 'content': tool_use_responses})
