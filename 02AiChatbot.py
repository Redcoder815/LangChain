from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

# 1. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 2. Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Always answer in 1 line."),
        ("human", "{input}")
    ]
)

chain = prompt | llm

# 3. Memory storage and session tracking setup
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        # store[session_id] = ChatMessageHistory()
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4. Wrap the chain with history tracking capabilities
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input"
)

# 5. Interactive command-line chatbot loop
print("AI Chatbot with Memory (type 'exit' to quit)\n")

session_config = {"configurable": {"session_id": "user_session_1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
        
    response = chain_with_memory.invoke(
        {"input": user_input},
        config=session_config
    )
    
    print(f"AI: {response.text}\n")
