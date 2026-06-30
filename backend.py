from streamlit import form
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
from typing import TypedDict
from dotenv import load_dotenv
import os
from typing import Optional, Annotated
load_dotenv()
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_aws.chat_models import ChatBedrock

global_retriever = None

def process_pdf(file_bytes):
    global global_retriever
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        file_path = tmp.name
        
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    global_retriever = vectorstore.as_retriever()
    return True

# OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
# OLLAMA_BASE_URL = "https://ollama.com"
# OLLAMA_MODEL = "gpt-oss:20b"

# client_kwargs = {}
# if OLLAMA_API_KEY and OLLAMA_BASE_URL.startswith("https://ollama.com"):
#     client_kwargs["headers"] = {
#         "Authorization": f"Bearer {OLLAMA_API_KEY}",
#     }

# model = ChatOllama(
#     model=OLLAMA_MODEL,
#     base_url=OLLAMA_BASE_URL,
#     temperature=0.5,
#     client_kwargs=client_kwargs,
#     async_client_kwargs=client_kwargs,
# )
model = ChatBedrock(
    model_id="qwen.qwen3-235b-a22b-2507-v1:0",
    region_name="ap-south-1",
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: ChatState):
    messages = state['messages']
    
    if global_retriever is not None and len(messages) > 0 and isinstance(messages[-1], HumanMessage):
        latest_query = messages[-1].content
        docs = global_retriever.invoke(latest_query)
        
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            sys_prompt = f"You are a helpful AI assistant. Use the following context from an uploaded PDF to answer the user's question if relevant.\n\nContext:\n{context}"
            
            # Temporary message list with the SystemMessage injected
            temp_messages = [SystemMessage(content=sys_prompt)] + messages
            response = model.invoke(temp_messages)
            return {"messages": [response]}
            
    response = model.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(ChatState)
checkpointer = InMemorySaver()
graph.add_node("chatbot",chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot",END)
workflow = graph.compile(checkpointer=checkpointer)

