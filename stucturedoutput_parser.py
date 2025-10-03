from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parser import StructuredOutputParser, ResponseSchema

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = 'fact_1',description = 'Fact 1 about topic'),
    ResponseSchema(name = 'fact_2',description = 'Fact 3 about topic'),
    ResponseSchema(name = 'fact_3',description = 'Fact 3 about topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)