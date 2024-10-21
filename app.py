import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_aws.llms.bedrock import BedrockLLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Define the State class
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the StateGraph and BedrockLLM
graph_builder = StateGraph(State)

# llm = BedrockLLM(model_id='amazon.titan-text-lite-v1')

llm = ChatGoogleGenerativeAI(model='gemini-pro')


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add nodes and edges to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Streamlit app setup
st.title("Chatbot")

# Initialize or update session state for conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
user_input = st.text_input("You:", key="user_input")

# Initialize a variable to store the current response
current_response = ""

# Button to send the message
if st.button("Send") or user_input:
    if user_input.lower() in ["quit", "exit", "q"]:
        current_response = "Goodbye!"
        st.session_state['history'].append(("User", user_input))
        st.session_state['history'].append(("Assistant", current_response))
    else:
        st.session_state['history'].append(("User", user_input))
        for event in graph.stream({"messages": ["user", user_input]}):
            for value in event.values():
                current_response = value["messages"][-1]
                st.session_state['history'].append(("Assistant", current_response))

# Display the response to the current question
if current_response:
    st.subheader("Response to Your Question")
    st.text(current_response)

# Display conversation history under "Chat History" heading
st.subheader("Chat History")
for speaker, message in st.session_state['history']:
    st.text(f"{speaker}: {message}")

