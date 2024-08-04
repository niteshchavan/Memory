import re
from flask import Flask, request, render_template, jsonify
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


embedding_function = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
llm = ChatOllama(model="qwen2:0.5b")



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a bot your name is Alice you should reply in 100 words or less"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{context}\n\nQ: {question}\nA:"),
    ]
)

chain = prompt | llm


chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)


def contains_url(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
    return url_pattern.search(text) is not None

def get_url(data):
    search = DuckDuckGoSearchResults(num_results=1)
    result = search.invoke(data)
    match = re.search(r'link:\s*(https?://[^\s\]]+)', result)
    if match:
        return match.group(1)
    else:
        return None

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    # Extract text from all <p> tags
    paragraphs = soup.find_all('p')
    text = "\n\n".join(p.get_text() for p in paragraphs)
    # Clean the text
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Remove multiple newlines
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def create_embeddings(url):
    try: 
        loader = RecursiveUrlLoader(url, extractor=bs4_extractor)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
        )
        docs = loader.load_and_split(text_splitter=text_splitter)
        docs = filter_complex_metadata(docs)
        Chroma.from_documents(docs, embedding_function, persist_directory="chroma_db")
        try:
            
            first_doc_title = docs[0].metadata.get('title', 'No title available')
            return first_doc_title
        except Exception as e:
            print(e)
            return 'Failed to get Title'
    except Exception as e:
        print(e)
        return 'Failed to process'

def retriever(data):
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.1,
        },
    )
    print("Retriver input:", data)
    documents = retriever.invoke(data)    
    return documents

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/query', methods=['POST'])
def query():
    try: 
        json_data = request.get_json()
        data = json_data.get('query_text', '')
        
        # Check if Data has a URL
        if contains_url(data):
            print("URL found")
            url = data
            result = create_embeddings(url)
            return jsonify({'message': result}), 200
        
        #IF url is not present
        else:
            #Check if chromad db has data or else search online for data
            retrieved_data = retriever(data)
            if retrieved_data:
                print(retrieved_data)
                context = "\n\n".join([doc.page_content for doc in retrieved_data])
                config = {"configurable": {"session_id": "1"}}
                response = chain_with_history.invoke({"question": data, "context": context }, config=config)
                return jsonify({'message': response.content}), 200
            else:
                print("No data found, searching online")
                url = get_url(data)
                if url:
                    result = create_embeddings(url)
                    print("Data Added: ", result)
                    retrieved_data = retriever(data)
                    context = "\n\n".join([doc.page_content for doc in retrieved_data])
                    config = {"configurable": {"session_id": "1"}}
                    response = chain_with_history.invoke({"question": data, "context": context }, config=config)
                    return jsonify({'message': response.content}), 200
                else:
                    return jsonify({'message': 'No URL found and no data available'}), 404

            

    except Exception as e:
        print(e)
        return jsonify({'message': 'Failed to process'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
