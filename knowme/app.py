import streamlit as st
from knowme.agent import KnowMeAgent
from dotenv import load_dotenv
from knowme.load_chains import load_site_answer_chain, load_cv_answer_chain
import zipfile
from pathlib import Path
import os

# Load the environment variables
load_dotenv()
st.set_page_config(layout="wide")

notion_folderpath = None
notion_vectorstore = None
cv_filename = None
cv_vectorstore = None
site_chain = None
cv_chain = None


DATA_DIR = os.environ["DATA_DIR"]
STORES_DIR = os.environ["STORES_DIR"]
#######################################################################################
# Information about the app
#######################################################################################
# Write what the app does
with st.sidebar:
    st.markdown("# Know Me")
    st.markdown("## Want to ask question about a person?")
    st.markdown(
        "This app helps you chat and obtain information about a person using ChatGPT."
    )

    st.markdown("## Here is what the app does:")
    st.markdown("1. Answers questions from their Notion page")
    st.markdown("2. Answers questions from their CV")
    st.markdown("3. You can either chat with their website or from the CV")
    st.markdown(
        "4. There is also an agent that automatically choses the approriate source"
    )
    st.markdown(
        "5. For example; some information is present in the CV and some in the website"
    )

#######################################################################################
# Just get the input for the notion zip folder and the CV
#######################################################################################
notion_column, cv_column = st.columns(2, gap="large")


with notion_column:
    st.markdown("### Upload your Notion folder")
    notion_folder = st.file_uploader("Upload the .zip Notion page")
    with st.expander("Tip: How to export your Notion Page?"):
        st.markdown("""How to export your notion page to a .zip folder? Follow these steps:
                    1. 
                    """)

    if notion_folder:
        if notion_folder.type == "application/zip":
            with zipfile.ZipFile(notion_folder, "r") as z:
                # Using the root folder as the folder to
                z.extractall(DATA_DIR)

            notion_folderpath = Path(f"{DATA_DIR}/{notion_folder.name}").with_suffix("")

            # create the vector path if it not provided
            notion_vectorstore = Path(f"{STORES_DIR}/{notion_folder.name}_vectorstore")
            notion_vectorstore = str(notion_vectorstore)

            st.success("Extracted the zip folder. Ready to answer your questions")

        else:
            st.error("Please upload a .zip folder. Read the tip above if needed")

with cv_column:
    st.markdown("### Upload your CV")
    cv_file = st.file_uploader("Upload your CV in .pdf format")

    if cv_file:
        filename = cv_file.name
        filename = Path(filename)

        if filename.suffix == ".pdf":
            with open(f"{DATA_DIR}/{filename}", "wb") as fp:
                fp.write(cv_file.getvalue())
            st.success(
                "Successfully uploaded {filename}. Ready to answer your questions"
            )
        else:
            st.error("Please upload a .pdf file. ")

        cv_filename = f"{DATA_DIR}/{filename}"

        # create the cv vectorstore
        cv_vectorstore = Path(f"{STORES_DIR}/{cv_file.name}_vectorstore")
        cv_vectorstore = str(cv_vectorstore)

st.divider()


# Show options to chose from.
# Does the user want answer from the site or from the CV or the agent
# The agent should be the first option


def reset_session_state():
    st.session_state.messages = []


chat_option = st.selectbox(
    "What do you want to chat with?",
    ("website", "cv", "agent"),
    format_func=lambda str: str.capitalize(),
    on_change=reset_session_state,
)

# define bools to decide which chains to load
is_load_site_chain = False
is_load_cv_chain = False
is_load_agent = False

if chat_option == "agent":
    st.warning("Agent is a Experimental Feature. Latency is high", icon="⚠️")
    if notion_folderpath is None or notion_vectorstore is None:
        st.error("Please upload your Notion website zip folder", icon="📁")

    if cv_filename is None or cv_vectorstore is None:
        st.error("Please upload your CV", icon="📑")

    is_load_agent = True

elif chat_option == "website":
    if notion_folderpath is None or notion_vectorstore is None:
        st.error("Please upload the Notion website zip folder", icon="📁")
    else:
        is_load_site_chain = True

elif chat_option == "cv":
    if cv_filename is None or cv_vectorstore is None:
        st.error("Please upload your CV", icon="📑")
    else:
        is_load_cv_chain = True


if is_load_site_chain:
    with st.spinner("Loading the Site Chat Agent 🤖"):
        chat = load_site_answer_chain(
            notion_folderpath=notion_folderpath,
            embedding_store_directory=notion_vectorstore,
        )

elif is_load_cv_chain:
    with st.spinner("Loading the CV Chat Agent 🤖"):
        chat = load_cv_answer_chain(
            cv_filepath=cv_filename, embedding_store_directory=cv_vectorstore
        )

elif is_load_agent:
    if is_load_site_chain:
        site_chain = load_site_answer_chain(
            notion_folderpath=notion_folderpath,
            embedding_store_directory=notion_vectorstore,
        )

    if is_load_cv_chain:
        cv_chain = load_cv_answer_chain(
            cv_filepath=cv_filename, embedding_store_directory=cv_vectorstore
        )

    if site_chain is not None and cv_chain is not None:
        chat = KnowMeAgent(website_chain=site_chain, cv_chain=cv_chain)


# This displays the chat history after refreshing
# The history is stored in session_state object
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        with st.expander(label=f"{message['role'].capitalize()} says:", expanded=True):
            st.markdown(message["content"])


if prompt := st.chat_input("What do you want to know about me?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        with st.expander(label="User says: ", expanded=True):
            st.markdown(prompt)


# This steps through the chain stream
# and yields those chunks that have an answer with them
def chunk_generator(stream):
    for chunk in stream:
        if chunk is None:
            break
        if chunk.get("answer"):
            yield chunk["answer"]


if prompt:
    with st.chat_message("assistant"):
        # the answer here is a stream
        with st.spinner("Our Agent is working hard to find an answer"):
            answer = chat.chat_stream(prompt, session_id="abc")
            with st.expander("Assistant Says: ", expanded=True):
                # The last message is the output of the write_stream function
                final_answer = st.write_stream(chunk_generator(answer))
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )
