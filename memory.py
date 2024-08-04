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
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults
import langchain
import re


langchain.debug = True

app = Flask(__name__)

embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

llm = ChatOllama(model="qwen2:0.5b")

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

def create_embedings(query_text):
        print(query_text)
        loader = RecursiveUrlLoader(query_text, extractor=bs4_extractor)
            
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
            )
            
        docs = loader.load_and_split(text_splitter=text_splitter)

        first_doc_title = docs[0].metadata.get('title', 'No title available')

        docs = filter_complex_metadata(docs)
        print(first_doc_title)
        Chroma.from_documents(docs, embedding_function, persist_directory="chroma_db")

        return first_doc_title



store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

import warnings

def create_retriever(query_text):

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")


        try:
            retriever = db.as_retriever(
                    search_type="similarity_score_threshold",
                        search_kwargs={
                            "k": 5,
                            "score_threshold": 0.1,
                        },
                )
            
            history_aware_retriever = create_history_aware_retriever(
            llm, retriever , contextualize_q_prompt )

            


            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            
            
            result = conversational_rag_chain.invoke(
                    {"input": query_text },
                    config={
                        "configurable": {"session_id": "abc123"}
                    },  # constructs a key "abc123" in `store`.
                    
                    )["answer"],

            
                    # Check for warnings
            for warning in w:
                    if "No relevant docs were retrieved" in str(warning.message):
                        print("No relevant documents were found with the given relevance score threshold.")
                        result = search(query_text)
                        return result
                    
            return result
            
                            
            
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Failed to retrieve documents."

def search(query_text):
    search = DuckDuckGoSearchResults(num_results=1)
    result = search.invoke(query_text)
    match = re.search(r'link:\s*(https?://[^\s\]]+)', result)
    if match:
        link_value = match.group(1)
        first_doc_title = create_embedings(link_value)
        print(first_doc_title)
        result = create_retriever(query_text)
        return result
    else:
        return "No search results found."


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
            return jsonify({'message': f'AI: {first_doc_title}'}), 200
        
        else:
            result = create_retriever(query_text)
            return jsonify({'message': f'AI: {result}'}), 200
         

    except Exception as e:
        print(e)
        return jsonify({'message': 'Failed to process'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
