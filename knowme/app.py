import streamlit as st
import os
from knowme.ingest import NotionIngestor
from knowme.embedder import ChromaEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from knowme.knowme_chat import KnowmeChat


st.title("Know Me")
# Write more about what the app does

# Get the key from the user
openai_key = st.sidebar.text_input("Enter Your OpenAI Key", type="password")


# TODO: Move these to a cache call that sets up everything only once.
notion_folderpath = "../abhinav_notion"

ingestor = NotionIngestor(notion_folderpath)
texts, metadatas = ingestor.ingest()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

embedding_function = OpenAIEmbeddings()

store = ChromaEmbedder(
    text_splitter,
    embedding_function=embedding_function,
    embedding_store_directory="../notion_site_store",
)

chromastore = store.store_embeddings(texts, metadatas)
llm = ChatOpenAI(model="gpt-4")


chat = KnowmeChat(llm, chromastore)


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
        airesponse = chat.chat(str(prompt), session_id="abc")
        st.session_state.messages.append({"role": "assistant", "content": airesponse})
        st.markdown(airesponse)
