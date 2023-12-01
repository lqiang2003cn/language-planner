from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores import ElasticsearchStore

collection_name = "available_example_index"
embedding = OpenAIEmbeddings()
vectorstore = ElasticsearchStore(
    es_url="http://192.168.50.66:9200",
    index_name=collection_name,
    embedding=embedding
)

task = 'Make breakfast'
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})
retrieved_docs = retriever.invoke(task)[0]
print(retrieved_docs)

curr_prompt = f'{retrieved_docs.page_content}\n\nTask: {task}'
