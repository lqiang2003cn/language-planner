import openai
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore

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

loader = JSONLoader(
    file_path='available_examples.json',
    jq_schema='.[]',
    text_content=True
)
data = loader.load()
example = data[retrieved_docs.metadata['seq_num'] - 1]
curr_prompt = f'{example.page_content}\n\nTask: {task}'
print(curr_prompt)

MAX_STEPS = 20
# "max_tokens": 10,
# "temperature": 0.6,
# "top_p": 0.9,
# "n": 10,
# "logprobs": 1,
# "presence_penalty": 0.5,
# "frequency_penalty": 0.3,
# "stop": '\n'

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=curr_prompt,

    logprobs=1,
    max_tokens=10,
    echo=True,
    temperature=0.6,
    stop='\n',

    # max_tokens=10,
    # temperature=0.6,
    # top_p=0.9,
    # n=10,
    # logprobs=1,
    # presence_penalty=0.5,
    # frequency_penalty=0.3,
    # stop='\n',
)

# generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
# # calculate mean log prob across tokens
# mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
# generated_samples = [sample.strip().lower() for sample in generated_samples]
# return generated_samples, mean_log_probs
#
# ["choices"][0]["logprobs"]["token_logprobs"]
print(response)
