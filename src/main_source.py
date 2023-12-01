from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from langchain.indexes import SQLRecordManager, index

collection_name = "source_test_index"
embedding = OpenAIEmbeddings()
vectorstore = ElasticsearchStore(
    es_url="http://192.168.50.66:9200",
    index_name=collection_name,
    embedding=embedding
)
namespace = f"elasticsearch/{collection_name}"
sql = f"sqlite:///{collection_name}.sql"
record_manager = SQLRecordManager(namespace, db_url=sql)
record_manager.create_schema()


def clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


clear()

doc1 = Document(page_content="kitty kitty kitty kitty kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy doggy the doggy", metadata={"source": "doggy.txt"})

new_docs = CharacterTextSplitter(separator="t", keep_separator=True, chunk_size=12, chunk_overlap=2).split_documents([doc1, doc2])
print(new_docs)
print(index(
    new_docs,
    record_manager,
    vectorstore,
    cleanup=None,
    source_id_key="source",
))

changed_doggy_docs = [
    Document(page_content="woof woof", metadata={"source": "kitty.txt"}),
    Document(page_content="woof woof woof", metadata={"source": "kitty.txt"}),
]
print(index(
    changed_doggy_docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
))