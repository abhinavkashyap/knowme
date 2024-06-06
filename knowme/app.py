import streamlit as st
import os
from dotenv import load_dotenv
from knowme.load_chains import load_site_answer_chain

# Load the environment variables
load_dotenv()


st.title("Know Me")
# Write more about what the app does

# Get the key from the user
openai_key = os.environ["OPENAI_API_KEY"]


# TODO: Move these to a cache call that sets up everything only once.
notion_folderpath = os.environ["NOTION_FOLDER"]
site_vectorstore = os.environ["NOTION_SITE_VECTORSTORE"]


chat = load_site_answer_chain(
    notion_folderpath=notion_folderpath,
    embedding_store_directory=site_vectorstore,
)


# this displays the chat history after refreshing
# The history is stored in session_state object
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What do you want to know about me?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

if prompt:
    with st.chat_message("assistant"):
        answer = chat.chat(prompt, session_id="abc")
        with st.expander("Know Me Says: ", expanded=True):
            st.markdown(answer)
