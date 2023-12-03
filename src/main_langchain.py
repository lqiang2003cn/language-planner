from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from openlm.llm import OpenAI

llm = OpenAI(
    temperature=0.6
    # max_tokens=10,
    # n=10,
    # model_kwargs={
    #     "top_p": 0.9,
    #     "logprobs": 1,
    #     "presence_penalty": 0.5,
    #     "frequency_penalty": 0.3,
    # }
)

prompt = PromptTemplate.from_template("Task: Cook some food\n\nTask: Make breakfast")
o = llm(prompt.template)
# chain = prompt | llm
# p = chain.({})
print(o)
