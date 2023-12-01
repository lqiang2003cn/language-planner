import json

from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.chroma import Chroma

underlying_embeddings = OpenAIEmbeddings()

with open('available_actions_min.json', 'r') as f:
    action_list = json.load(f)

with open('available_examples_min.json', 'r') as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]

fs = LocalFileStore("/home/lq/lq_projects/seamless_communication/language-planner/src/vectordb")
action_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs, namespace=underlying_embeddings.model)
# action_list_embeds = action_embedder.embed_documents(action_list)
# example_task_list_embeds = action_embedder.embed_documents(example_task_list)

action_db = Chroma.from_texts(action_list, action_embedder)
retriever = action_db.as_retriever()
retrieved_docs = retriever.invoke("beer")
print(retrieved_docs)