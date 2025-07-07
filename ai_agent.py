import os
from dotenv import load_dotenv
load_dotenv()

# api keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GEMINI_API_KEY =os.getenv("GEMINI_API_KEY")

# setup llm and tools
# -- important libraris
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults


search_tool=TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):

    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "Gemini":
        llm = ChatGoogleGenerativeAI(model=llm_id,
                                     google_api_key=GEMINI_API_KEY, # <-- Corrected parameter name
                                    convert_system_message_to_human=True 
                                    )
    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt
        )

    state = {"messages":query}
    response = agent.invoke(state)
    messages = response.get("messages")
    generated_messages = [message.content for message in messages if isinstance(message,AIMessage)]
    return generated_messages[-1]

