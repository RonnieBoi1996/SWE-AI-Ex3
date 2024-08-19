import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI
from duckduckgo_search import DDGS

load_dotenv()

model = AzureOpenAI(
    azure_endpoint='https://openaifor3267.openai.azure.com/',
    azure_deployment='gpt4',
    api_version='2024-02-01',
    api_key=os.getenv('OPENAI_API_KEY'),,
)


def extract_entities_from_file(file_name: str, entity_type: str) -> str:
    """
    This function extract entities of a given type from a given text file.
    The function is given a text file and an 'entity type' (e.g. 'student',
    'hobby', 'city' or any other type of entity). It returns a string representing
    a list of entities of that type in the file (e.g. “[‘Haifa’, ‘Tel Aviv’]”)
    """
    with open(file_name, 'r') as file:
        text = file.read()
    messages = [
        {
            'role': 'system',
            'content': "You will be given a text file. Extract the entities of the given type from the file and return them as a python list.",
        },
        {
            'role': 'system',
            'content': "Do NOT speicfy the entity type. Just return the list and nothing else. Don't write anything that is not part of the list.",
        },
        {
            'role': 'system',
            'content': "For example, if the entity type is 'city', you should extract all the cities from the file.",
        },
        {
            'role': 'system',
            'content': "Another example: if the entity type is 'food', you should extract all the foods from the file.",
        },
        {
            'role': 'user',
            'content': f'The entity type is:\n{entity_type}'
        },
        {
            'role': 'user',
            'content': f'The text:\n{text}'
        },
    ]

    return model.chat.completions.create(messages=messages, model="gpt4").choices[0].message.content


def Internet_search_attribute(entity: str, attribute: str) -> str:
    """
    Uses the DuckDuckGo search engine to find the attribute of an entity.
    It then asks the LLM to review the search results and return a JSON
    structure of the form:

    {"entity": {entity}, "{attribute}": Answer}.

    Example answer:

    { "city": ”Tel Aviv", "population": 4,400,000 }
    """
    keywords = f'{entity} {attribute}'
    search_result = DDGS().text(keywords=keywords, max_results=1)
    messages = [
        {
            'role': 'system',
            'content': f"""
You will be given a text with information about the {attribute} of {entity}.
Retrieve this information and return it in the following JSON format:
"entity":{entity}, "{attribute}": <The information you found>"""
        },
        {
            'role': 'user',
            'content': f"The text:\n{search_result[0]['body']}"
        }
    ]

    return model.chat.completions.create(messages=messages, model="gpt4").choices[0].message.content


def generate_analysis_program(query_description: str, input_file: str, columns: str,
                              row_example: str, output_file: str) -> str:
    """
    This function takes a query_description, a csv input_file, the column names of the csv file, 
    another example of a row within the csv file and an output_file.

    query_description describes an analysis of the csv input_file to be
    performed. This function generates a Python program to perform the analysis.
    The program writes the output, a JSON object, to the output_file.
    generate_analysis_program returns the file name of the generated Python code.
    """

    messages = [
        {
            'role': 'system',
            'content': "You are an expert python coder that likes using the pandas library."
        },
        {
            'role': 'system',
            'content': "You will be given a description of an analysis to be performed on a data file."
        },
        {
            'role': 'system',
            'content': "The program you generate should output the query results as a JSON object."
        },
        {
            'role': 'system',
            'content': "Only respond with code and comments. Do NOT write anything that is not code or comments. I'm serious. Don't you dare write anything that is not code. "
        },
        {
            'role': 'user',
            'content': f"Analysis request:\n{query_description}"
        },
        {
            'role': 'user',
            'content': f"The data file to analyze:\n{input_file}"
        },
        {
            'role': 'user',
            'content': f"The columns of the data file:\n{columns}"
        },
        {
            'role': 'user',
            'content': f"Row example:\n{row_example}"
        },
        {
            'role': 'user',
            'content': f"The file to which the program should output to:\n{output_file}"
        },
    ]

    res = model.chat.completions.create(
        messages=messages, model="gpt4", temperature=0.6).choices[0].message.content
    # Since the LLM responds with a code block, we need to remove the code block syntax
    res = res.removeprefix('```python')
    res = res.removesuffix('```')
    return res.strip()


def execute_Python_program(program_file: str) -> str:
    """
    This function executes the program in file program_file. It returns
    either a message that the program executed successfully, or it
    returns a message that the program did not execute successfully
    and provides error messages.
    """
    with open(program_file, 'r') as file:
        code = file.read()

    try:
        exec(code)
        return "Successfully executed."
    except Exception as e:
        return f"Error: {str(e)}"


def write_file(file_content: str, output_file: str):
    """
    This function writes the file's content to the given file path.
    """
    with open(output_file, "w") as file:
        file.write(file_content)


def get_tools_list():
    func_description = [
        extract_entities_from_file.__doc__,
        Internet_search_attribute.__doc__,
        generate_analysis_program.__doc__,
        execute_Python_program.__doc__,
        write_file.__doc__,
    ]
    return [
        {  # extract_entities_from_file
            "type": "function",
            "function": {
                "name": "extract_entities_from_file",
                "description": f"{func_description[0]}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "The name of the file to extract entities from.",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "The entity type, e.g. city, hobby, name etc...",
                        },
                    },
                    "required": ["file_name", "entity_type"],
                }
            }
        },
        {  # Internet_search_attribute
            "type": "function",
            "function": {
                "name": "Internet_search_attribute",
                "description": f'{func_description[1]}',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "The entity to search for.",
                        },
                        "attribute": {
                            "type": "string",
                            "description": "The attribute of the entity to search for.",
                        },
                    },
                    "required": ["entity", "attribute"],
                }
            }
        },
        {  # generate_analysis_program
            "type": "function",
            "function": {
                "name": "Internet_search_attribute",
                "description": f'{func_description[2]}',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_description": {
                            "type": "string",
                            "description": "Description of the analysis query to be performed on the file.",
                        },
                        "input_file": {
                            "type": "string",
                            "description": "The CSV file to analyze.",
                        },
                        "columns": {
                            "type": "string",
                            "description": "Names of the columns in the CSV file.",
                        },
                        "row_example": {
                            "type": "string",
                            "description": "An example row from the CSV file.",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "The file to which the program writes the output to.",
                        },
                    },
                    "required": ["analysis_request", "input_file", "columns", "row_example", "output_file"],
                }
            }
        },
        {  # execute_Python_program
            "type": "function",
            "function": {
                "name": "execute_Python_program",
                "description": f'{func_description[3]}',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "program_file": {
                            "type": "string",
                            "description": "The Python file to execute.",
                        },
                    },
                    "required": ["program_file"],
                }
            }
        },
        {  # write_file
            "type": "function",
            "function": {
                "name": "write_file",
                "description": f'{func_description[4]}',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_content": {
                            "type": "string",
                            "description": "The content to be written into the file.",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "The file to be written into.",
                        },
                    },
                    "required": ["entity", "attribute"],
                }
            }
        },
    ]
