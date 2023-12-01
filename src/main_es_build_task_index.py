from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.vectorstores import ElasticsearchStore

loader = JSONLoader(
    file_path='available_examples.json',
    jq_schema='.[]',
    text_content=True
)
data = loader.load()
for d in data:
    d.page_content = d.page_content.split('\n')[0]

collection_name = "available_example_index"
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
    print(index([], record_manager, vectorstore, cleanup="full", source_id_key="seq_num"))

# clear()

print(index(
    data,
    record_manager,
    vectorstore,
    batch_size=1000,
    cleanup=None,
    source_id_key="seq_num",
))

# print(index(
#     data_10_20,
#     record_manager,
#     vectorstore,
#     cleanup=None,
#     source_id_key="source",
# ))

# print(index(
#     data_0_20,
#     record_manager,
#     vectorstore,
#     cleanup=None,
#     source_id_key="seq_num",
# ))
#
# data[10].page_content = 'good morning'
#
# print(index(
#     data[10:11],
#     record_manager,
#     vectorstore,
#     cleanup='incremental',
#     source_id_key="seq_num",
# ))

# print(index(
#     data_200,
#     record_manager,
#     vectorstore,
#     cleanup=None,
#     source_id_key="source",
# ))

# print(index(
#     data_10_200,
#     record_manager,
#     vectorstore,
#     cleanup='incremental',
#     source_id_key="source",
# ))
