import json
import os
import sys

from dotenv import load_dotenv
from tools import *
from openai import AzureOpenAI


def get_params(params_as_obj):
    params = ""
    for key in params_as_obj:
        params += f'Parameter {key}='
        params += f'{params_as_obj[key]}'[:50]  # Limit to 50 characters
        params += ', '

    return params


def main():
    # Extract query and file resources from input file
    with open(sys.argv[1], 'r') as file:
        input_file = json.load(file)
    with open(input_file['query_name'], 'r') as file:
        query = file.read()
    file_resources = input_file['file_resources']

    load_dotenv()

    # The LLM
    model = AzureOpenAI(
        azure_endpoint='https://openaifor3267.openai.azure.com/',
        azure_deployment='gpt4',
        api_version='2024-02-01',
        api_key='os.getenv('OPENAI_API_KEY'),',
    )

    # Initial prompt
    messages = [
        {
            'role': 'system',
            'content': "You are a helpful AI assitant."
        },
        {
            'role': 'system',
            'content': "You need to perform a task for the user."
        },
        {
            'role': 'system',
            'content': "You might also recieve a list of file resources related to the task."
        },
        {
            'role': 'system',
            'content': "You might need to use tools to complete the task."
        },
        {
            'role': 'system',
            'content': "If a program that you wrote fails to execute, you should regenerate the program."
        },
        {
            'role': 'system',
            'content': "When you are done, you should say 'Done.'. This is the only way you should respond if you are done."
        },
        {
            'role': 'user',
            'content': f"The task is:\n{query}"
        },
        {
            'role': 'user',
            'content': f"The file resources are:\n{file_resources}"
        },
    ]

    # Agent logic
    tools = get_tools_list()
    LLM_invocations_counter = 0
    tool_calls_counter = 0
    log_file_content = ""
    while ((LLM_invocations_counter < 10) and (tool_calls_counter < 10)):
        res = model.chat.completions.create(
            model="gpt4",
            messages=messages,
            tools=tools,
        ).choices[0].message
        LLM_invocations_counter += 1  # Increment invocation counter
        if (res.content == "Done."):  # Exit loop if task is done
            break
        elif (res.tool_calls != None):  # Execute tool if tool call is present
            tool_name = res.tool_calls[0].function.name
            tool_args = json.loads(res.tool_calls[0].function.arguments)
            log_file_content += f"**Entering agent {tool_name}**\n"
            # Add tool parameters to log file
            log_file_content += f'{get_params(tool_args)}\n'
            tool_result = eval(tool_name)(**tool_args)
            messages.append({
                'role': 'user',
                'content': f"Tool {tool_name} returned: {tool_result}"
            })
            if (tool_name == "execute_Python_program"):  # Add execution result to log file
                log_file_content += f"{tool_result}\n"
            log_file_content += f"**Leaving agent {tool_name}**\n\n"
            tool_calls_counter += 1  # Increment tool calls counter

    # Create log file
    with open(f'log_{input_file['query_name']}', 'w') as file:
        file.write(log_file_content)


if __name__ == '__main__':
    main()
