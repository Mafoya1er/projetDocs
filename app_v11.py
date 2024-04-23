from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
print(chain.invoke({"text": "Parle moi de Groq."})["content"])

llama_clou_api_key="llx-15it5aindiMHkKwbasNKUWxMp7y6YsCBZouF6mGz4LeseMAz"