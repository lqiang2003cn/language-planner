from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores import ElasticsearchStore

collection_name = "available_action_index"
embedding = OpenAIEmbeddings()
vectorstore = ElasticsearchStore(
    es_url="http://192.168.50.66:9200",
    index_name=collection_name,
    embedding=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
retrieved_docs = retriever.invoke("eat apple")
print(retrieved_docs)

