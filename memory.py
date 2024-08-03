from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import SQLChatMessageHistory


import re

app = Flask(__name__)

def contains_url(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
    return url_pattern.search(text) is not None


# Extractor function to get all text within <p> tags
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    paragraphs = soup.find_all('p') # Extract text from all <p> tags
    text = "\n\n".join(p.get_text() for p in paragraphs)
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple newlines
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

llm = ChatOllama(model="qwen2:0.5b")

def create_embedings(query_text):
        loader = RecursiveUrlLoader(query_text, extractor=bs4_extractor)
            
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
            )
            
        docs = loader.load_and_split(text_splitter=text_splitter)
            #response_message = docs[0].page_content
            #print(response_message)
            #print([doc.metadata for doc in docs])
        first_doc_title = docs[0].metadata.get('title', 'No title available')
            #print(first_doc_title)
        docs = filter_complex_metadata(docs)
        Chroma.from_documents(docs, embedding_function, persist_directory="chroma_db")
            #doc_len = len(docs)
            #print(doc_len)
        return first_doc_title



def create_retriver(query_text):
    retriever = db.as_retriever()

    return response





@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/query', methods=['POST'])
def query():
    try: 
        data = request.get_json()
        query_text = data.get('query_text', '')
        print(query_text)
        
        if contains_url(query_text):
            first_doc_title = create_embedings(query_text)
            return jsonify({'message': f'Title: {first_doc_title}'}), 200
        
        else:
            result = create_retriver(query_text)
            return jsonify({'message': f'Title: {result}'}), 200
         

    except Exception as e:
        print(e)
        return jsonify({'message': 'Failed to process'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
