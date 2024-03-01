# Importing necessary libraries
import gradio as gr
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Function to get vector store from a given URL
def getvecstore(url):
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vector store from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

# Function to get a context-aware retriever chain
def getcontext(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    # Define the prompt for context-aware retrieval
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

# Function to get a conversational RAG (Retrieval-Augmented Generation) chain
def getragchain(retriever_chain):
    llm = ChatOpenAI()

    # Define the prompt for conversational RAG
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # Create a chain for generating responses based on context
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Global variables
chat_history = [AIMessage(content="I am an assistant made to answer your questions?")]
vector_store = None

# Function to get response from the chatbot
def getresp(user_input, website_url):
    global chat_history, vector_store
    if vector_store is None:
        vector_store = getvecstore(website_url)

    retriever_chain = getcontext(vector_store)
    conversation_rag_chain = getragchain(retriever_chain)

    # Invoke the chain to get the response
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    # Update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    return response['answer']

# Function to interact with the chatbot through Gradio interface
def interact(Query, URL, OpenAI_Key):
    # Set the OpenAI key
    os.environ['OPENAI_API_KEY'] = OpenAI_Key
    # Get the response from the chatbot
    response = getresp(Query, URL)
    return response

# Create a Gradio interface
iface = gr.Interface(
    fn=interact,
    inputs=["text", "text", "text"],
    outputs="text",
    title="Chat with LLM",
    description="Interact with a chatbot based on your website URL. Also Provide your OpenAI key as well to get it working."
)

# Launch the Gradio interface
iface.launch()
