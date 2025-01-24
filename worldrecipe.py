import os
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Custom Search API setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Your Google API Key
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Your Custom Search Engine ID

def google_search(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
    response = requests.get(search_url)
    search_results = response.json().get("items", [])
    return [result["snippet"] for result in search_results]

# Step 1: Fetch Google Search results and process them into a vector store
def fetch_and_store_google_data(query):
    # Fetch data from Google
    search_results = google_search(query)
    docs = []

    # Process the retrieved search results
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    for result in search_results:
        # Convert each result to a Document object
        doc = Document(page_content=result)
        docs.extend(text_splitter.split_documents([doc]))

    # Create and save vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="db"
    )
    print("Vectorstore created and saved!")
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 2: Initialize the agent and set up the tools
def setup_agent(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    tools = [Tool(name="Document Retriever", func=retriever.get_relevant_documents, description="Fetch relevant documents.")]
    agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")
    return agent

# Streamlit User Interface
def app():
    st.title("AI-Powered Google Search Assistant")

    # User input: Query
    query = st.text_input("Ask a question:")

    if query:
        # Fetch and process data from Google based on the query
        st.write("Processing your query...")
        retriever = fetch_and_store_google_data(query)
        agent = setup_agent(retriever)
        
        # Use the agent to answer the query
        answer = agent.run(query)
        
        # Display the answer
        st.write("### Answer:")
        st.write(answer)

        # Display relevant documents fetched from Google
        st.write("### Relevant Google Search Results:")
        results = retriever.get_relevant_documents(query)
        for result in results:
            st.write(result.page_content)

# Run the app
if __name__ == "__main__":
    app()
