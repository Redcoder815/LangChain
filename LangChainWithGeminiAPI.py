from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    # google_api_key=google_api_key
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=gemini_model,
    tools=[get_weather],
    system_prompt="You are a weather assistance agent in 50 words"
)

# Pass the conversational sequence block into the agent graph
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like in Dhaka?"}
    ]
})

# Display the output directly
print(response["messages"][-1].content)