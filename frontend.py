#Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st
import requests
import time

st.set_page_config(page_title="Agentic AI Chatbot",layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt=st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.selectbox("Select Provider: ",["Groq","OpenAI"])
# provider=st.radio("Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search=st.checkbox("Allow Web Search")
user_query_input=st.text_area("Enter your query here: ", height=150, placeholder="Ask Anything!")

API_URL="http://127.0.0.1:8998/chat"

if st.button("Ask Agent"):
    if user_query_input.strip():
        with st.spinner("Thinking..."):
        #Step2: Connect with backend via URL

            payload = {
                "model_name": selected_model,
                "model_provider": provider,
                "system_prompt": system_prompt,
                "messages": [user_query_input],
                "allow_search": allow_web_search
            }

            response = requests.post(API_URL,json=payload)
            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    time.sleep(1)
                    st.write("Done")
                    st.subheader("Agent Response")
                    st.markdown(f"**Final Response:** {response_data}")
            