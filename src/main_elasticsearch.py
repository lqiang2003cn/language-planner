from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores import ElasticsearchStore


collection_name = "test_index"
embedding = OpenAIEmbeddings()
vectorstore = ElasticsearchStore(
    es_url="http://192.168.50.66:9200",
    index_name="test_index",
    embedding=embedding,
    # es_user="elastic",
    # es_password="changeme"
)
namespace = f"elasticsearch/{collection_name}"
record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
record_manager.create_schema()
doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})


def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


_clear()

print(index(
    [doc1, doc1, doc1, doc1, doc1],
    record_manager,
    vectorstore,
    cleanup=None,
    source_id_key="source",
))
retriever = vectorstore.as_retriever()
retrieved_docs = retriever.invoke("rocket")
print(retrieved_docs)

_clear()

print(index([doc1, doc2], record_manager, vectorstore, cleanup=None, source_id_key="source"))
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
retrieved_docs = retriever.invoke("cat")
print(retrieved_docs)

print(index([doc1, doc2], record_manager, vectorstore, cleanup=None, source_id_key="source"))
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
retrieved_docs = retriever.invoke("cat")
print(retrieved_docs)

_clear()

print(index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
))
print(index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
))
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
retrieved_docs = retriever.invoke("mouse")
print(retrieved_docs)
changed_doc_2 = Document(page_content="puppy", metadata={"source": "doggy.txt"})
print(index(
    [changed_doc_2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
))
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
retrieved_docs = retriever.invoke("beer")
print(retrieved_docs)

_clear()
all_docs = [doc1, doc2]
print(index(all_docs, record_manager, vectorstore, cleanup="full", source_id_key="source"))

del all_docs[0]
print(all_docs)
print(index(all_docs, record_manager, vectorstore, cleanup="full", source_id_key="source"))

