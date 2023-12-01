import json

from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.chroma import Chroma

underlying_embeddings = OpenAIEmbeddings()

with open('available_actions.json', 'r') as f:
    action_list = json.load(f)
fs = LocalFileStore("/home/lq/lq_projects/softwares_and_docs/caches/available_actions")
action_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs, namespace=underlying_embeddings.model)
# action_list_embeds = action_embedder.embed_documents(action_list)
print('finished embedding')
db = Chroma.from_texts(action_list, action_embedder)
retriever = db.as_retriever(search_kwargs={'k': 4})
retrieved_docs = retriever.invoke("banana")
print(retrieved_docs)
