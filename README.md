# AsktheWeb
Ask the Web is an application that retrieves data directly from the internet by scraping content.
It then uses a Retrieval-Augmented Generation (RAG) approach to generate intelligent and contextual answers to user queries without requiring any documents to be uploaded. This application integrates GenAI embeddings for vectorization and Gemini API for generating responses.

Features
*Scrape content directly from Web.
*Split and process scraped content into manageable chunks.
*Generate vector embeddings using GenAI embeddings.
*Build a vectorstore for efficient information retrieval.
*Answer user questions intelligently using Gemini API.
*No need to upload documents—works with live web data.


How It Works
1)Data Loading: Scrapes content from internet realted to the user query.
2)Data Processing: Splits content into smaller, manageable chunks for efficient embedding.
3)Embeddings Generation: Uses GenAI embeddings to generate vector representations of the chunks.
4)Vector Store: Stores embeddings in a vector database (e.g., Chroma) for quick similarity searches.
5)Question Answering: Combines relevant retrieved context with Gemini API to generate accurate and concise answers to user queries.

Installation
Prerequisites
Python 3.8 or later
Access to the following APIs:
Gemini API (for LLM text generation)
GenAI Embeddings API (for embedding generation)
Libraries:
  langchain
  chromadb
  requests
  beautifulsoup4
