from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

prompt = PromptTemplate(
    input_variables = ["purpose", "tone"],
    template = """
    write a {tone} email for following purpose: {purpose}
    keep it clear and concise.
    as short as possible.
"""
)

chain = prompt | llm

response = chain.invoke({
    "purpose":"Apologize for missing deadline",
    "tone":"Polite, Professional,apologetic"
})

print("---")
print(response.content[0]['text'])
print("---")