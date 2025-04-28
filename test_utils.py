from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import shutil
import os

def fetch_pdf_data(file_path: str) -> str:

    try:
        pdfreader = PdfReader(file_path)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        return raw_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def split_text(raw_text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> list:

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

def initialize_vectorstore(texts: list, embedding_model: str, directory: str = "./chromadb"):

    if os.path.exists(directory):
        shutil.rmtree(directory)

    embeddings = OllamaEmbeddings(model=embedding_model)


    if isinstance(texts[0], str):
        texts_content = texts
        metadatas = [{"source": "hris_manual"} for _ in texts_content]
    else:
        texts_content = [doc.page_content for doc in texts]
        metadatas = [{"source": doc.metadata.get("source", "hris_manual")} for doc in texts]

    vectorstore = Chroma(
        collection_name="hris-documents",
        embedding_function=embeddings,
        persist_directory=directory,
    )

    vectorstore.add_texts(texts=texts_content, metadatas=metadatas)
    print("Vectorstore initialized and populated.")
    return vectorstore

def initialize_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )

def initialize_qa_chain(retriever, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7):

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=1000
    )

    
    PROMPT = ChatPromptTemplate.from_messages([

    ("system", """You are an HRIS chatbot named HRIS Kurakani, developed on April 21, 2025, to assist users in navigating the HRIS system. Your responses should be friendly, professional, and clear, suitable for non-technical users. Follow these guidelines:

    - **HRIS Queries**: If the query is about the HRIS system (e.g., leave requests, manager roles), provide a concise, accurate answer using only the dataset. Use bullet points or numbered lists for steps or lists, and present the response in a detailed, ChatGPT-like format.
    - **Conversational Queries**: For off-topic or conversational questions, respond appropriately:
      - "Who are you?" or "What are you?": Answer, "I'm HRIS Kurakani, your friendly assistant for navigating the HRIS system. How can I help you today?"
      - "How are you?" or "How are you doing today?": Answer, "I'm doing great, thanks for asking! Ready to help with any HRIS questions you have."
      - Greetings (e.g., "Hi, I am Ram"): Answer, "Hello Ram, I'm HRIS Kurakani. Nice to meet you! How can I assist you with the HRIS system today?"
    - **User Info Queries**: If the query asks for previously provided information (e.g., "What was my name?"), use the conversation history. For example, if the user said "I am Prasanna," respond, "Your name is Prasanna."
    - **User Info Statements**: If the query provides user information (e.g., "I am Prasanna"), acknowledge it (e.g., "Nice to meet you, Prasanna! How can I assist you with the HRIS system today?") and store it for future reference.
    - **Unrelated Queries**: For other off-topic questions, respond, "I'm here to assist with HRIS system queries. Please ask a question about the HRIS system."
    - **No Data**: If the dataset lacks the answer, state, "The information is not available in the provided dataset."
    - **Context Awareness**: Use the conversation history to maintain context, especially for follow-up questions or user info.
    - **Avoid Repetition**: Do not repeat previous answers unless relevant.
    - Dont give introduction if question is related to the document.
    
    Context: {context}"""),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("user", "{input}")
    ])


    chain = create_stuff_documents_chain(
        llm = llm,
        prompt = PROMPT
    ) 


    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm = llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
        )
    
    return retrieval_chain


    # return RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=False,
    #     chain_type_kwargs={"prompt": PROMPT},
    # )

from langchain_core.messages import HumanMessage, AIMessage

def ask_question(qa_chain, query: str, chat_history: list = None) -> str:
    if chat_history is None:
        chat_history = []
    try:
      
        formatted_history = []
        for entry in chat_history:
            if entry.startswith("User: "):
                formatted_history.append(HumanMessage(content=entry[6:]))  
            elif entry.startswith("Bot: "):
                formatted_history.append(AIMessage(content=entry[5:])) 

        result = qa_chain.invoke({
            "input": query,
            "chat_history": formatted_history
        })

      
        answer = result.get("answer", "").strip()
        print(answer)
        return answer
    except Exception as e:
        return f"An error occurred: {e}"