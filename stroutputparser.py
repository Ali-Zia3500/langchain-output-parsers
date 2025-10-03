from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemmma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template='Write a 5 line Summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Cloud Brust in 2025'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)