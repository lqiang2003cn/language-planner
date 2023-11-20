import json

from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.chroma import Chroma

underlying_embeddings = OpenAIEmbeddings()

with open('available_actions.json', 'r') as f:
    action_list = json.load(f)

with open('available_examples.json', 'r') as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]

fs = LocalFileStore("/home/lq/lq_projects/softwares_and_docs/caches/available_actions/sample_cache/")
action_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs, namespace=underlying_embeddings.model)
embeddings = action_embedder.embed_documents(['make coffee', 'eat breakfast', 'eat lunch'])

db = Chroma.from_texts(action_list, action_embedder)
retriever = db.as_retriever()
retrieved_docs = retriever.invoke(
    "What did the president say about Ketanji Brown Jackson?"
)
