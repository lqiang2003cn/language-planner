import numpy as np
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document, SystemMessage
from langchain.vectorstores import ElasticsearchStore

task_index = "available_example_index"
embedding = OpenAIEmbeddings()
task_vectorstore = ElasticsearchStore(
    es_url="http://192.168.50.66:9200",
    index_name=task_index,
    embedding=embedding
)
task = 'make breakfast'
task_retriever = task_vectorstore.as_retriever(search_kwargs={'k': 1})
task_retriever_docs = task_retriever.invoke(task)[0]
print(task_retriever_docs)

loader = JSONLoader(
    file_path='available_examples.json',
    jq_schema='.[]',
    text_content=True
)
data = loader.load()
example = data[task_retriever_docs.metadata['seq_num'] - 1]
llm = ChatOpenAI(temperature=0)

system_prompt = """
You are an excellent household manager, you can break down a household task into a sequence of steps.
Given an example task and its corresponding steps, write the steps required for a new task.
Each step should has the form: '''Action Noun'''. There can be only one Action and one Noun in a command
The example task and its steps are:
{example}

Now, the new task is:
{new_task}  

Your response:"""

chat_template = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(system_prompt),
    ]
)

response = llm(chat_template.format_messages(example=example, new_task=task))
steps = response.content.split('\n')
print(steps)

action_index = "available_action_index"
embedding = OpenAIEmbeddings()
action_vectorstore = ElasticsearchStore(
    es_url="http://192.168.50.66:9200",
    index_name=action_index,
    embedding=embedding
)
action_retriever = action_vectorstore.as_retriever(search_kwargs={'k': 1})

action_prompt = """
Given a command, find verb and nouns in the command. 
Format your response as a comma seperated list, for example
'''verb, noun'''

The command is:
{action_example}

Your response:
"""

for s in steps:
    command = s.split(':')[1].strip()
    actions_retrieved_doc = action_retriever.invoke(command)[0]
    action_example = actions_retrieved_doc.page_content
    chat_template = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(action_prompt),
        ]
    )
    print(action_prompt.format(action_example=action_example))
    format_msg = chat_template.format_messages(action_example=action_example, new_command=command)
    response = llm(format_msg)
    print("Example:", action_example, "Rewrite:", response.content)





