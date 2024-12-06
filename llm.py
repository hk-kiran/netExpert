from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings


template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="llama3.2")
chain = prompt | model

response = chain.invoke({"question": "What is LangChain?"})
embeds = embeddings.embed_query("Hello, world!")
print(response)
print(embeds)