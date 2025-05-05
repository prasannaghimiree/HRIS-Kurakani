from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import shutil
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_pdf_data(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        pdfreader = PdfReader(file_path)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                content = ' '.join(content.split())  
                raw_text += content + '\n'
        logger.info(f"Extracted text length: {len(raw_text)} characters")
        return raw_text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return None

def split_text(raw_text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list:
    """Split text into chunks for vectorstore."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(raw_text)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def initialize_vectorstore(texts: list, embedding_model: str, directory: str = "./chromadb") -> Chroma:
    """Initialize and populate Chroma vectorstore with text embeddings."""
    if os.path.exists(directory):
        shutil.rmtree(directory)

    embeddings = OllamaEmbeddings(model=embedding_model)

    vectorstore = Chroma(
        collection_name="hris-documents",
        embedding_function=embeddings,
        persist_directory=directory,
    )

    texts_content = texts if isinstance(texts[0], str) else [doc.page_content for doc in texts]
    metadatas = [{"source": "hris_manual"}] * len(texts_content)

    vectorstore.add_texts(texts=texts_content, metadatas=metadatas)
    logger.info("Vectorstore initialized and populated.")
    return vectorstore

def initialize_retriever(vectorstore):
    """Initialize MMR-based retriever."""
    # return vectorstore.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    # )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever


def initialize_qa_chain(retriever, api_key: str, model: str = "gemini-2.0-flash", temperature: float = 0.7):
    """Initialize QA chain with history-aware retrieval."""
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=1000
    )

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are HRIS Kurakani, an HRIS chatbot developed on April 21, 2025, to assist users in navigating the HRIS system. Your responses should be friendly, professional, and clear, suitable for non-technical users. Follow these guidelines:
        - **HRIS Queries**: For queries about the HRIS system (e.g., leave requests, manager roles), provide a concise, accurate answer based only on the provided dataset. Use bullet points or numbered lists for steps or lists. Do not introduce yourself; directly provide the answer.
        - **Conversational Queries**:
            - "Who are you?" or "What are you?": Answer, "I'm HRIS Kurakani, your friendly assistant for navigating the HRIS system. How can I help you today?"
            - "How are you?" or "How are you doing today?": Answer, "I'm doing great, thanks for asking! Ready to help with any HRIS questions you have."
        - **Greetings** (e.g., "Hi, I am Ram"): Answer, "Hello Ram, I'm HRIS Kurakani. Nice to meet you! How can I assist you with the HRIS system today?"
        - **User Info Queries**: For queries about previously provided information (e.g., "What was my name?"), use conversation history. If the user said "I am Prasanna," respond, "Your name is Prasanna."
        - **User Info Statements**: For statements providing user info (e.g., "I am Prasanna"), acknowledge it (e.g., "Nice to meet you, Prasanna! How can I assist you with the HRIS system today?") and store it.
        - **Unrelated Queries**: For off-topic questions, respond, "I'm here to assist with HRIS system queries. Please ask a question about the HRIS system."
        - **No Data**: If the dataset lacks the answer, state, "The information is not available in the provided dataset."
        - **Context Awareness**: Use conversation history to maintain context for follow-up questions or user info. Prioritize history for queries like name recall or follow-ups.
        - **Avoid Repetition**: Do not repeat previous answers unless relevant.
        - **Ambiguous or Confusing Questions**
            - If the user asks something ambiguous or unclear, respond politely and professionally with:  
            - "I'm not sure how to help with that specific question. Could you please clarify what you meant?"  
            - "If you're facing any difficulty in navigating the system or have questions not covered here, please contact:  
                - Mr. Rupesh Malla at +977 9851225672  
                - Mr. Rohan Rai at +977 9841337041"
        

        Context: {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the conversation history and current query, generate a precise search query to retrieve relevant HRIS document content. Include user-specific context (e.g., names) and HRIS-specific terms.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(history_aware_retriever, chain)
    return retrieval_chain

def ask_question(qa_chain, query: str, chat_history: list = None) -> str:
    """Process a query with conversation history."""
    if chat_history is None:
        chat_history = []
    try:
        formatted_history = []
        for entry in chat_history:
            if entry.startswith("User: "):
                formatted_history.append(HumanMessage(content=entry[6:]))
            elif entry.startswith("Bot: "):
                formatted_history.append(AIMessage(content=entry[5:]))
        
        logger.debug(f"Processing query: {query}\nFormatted history: {formatted_history}")

        result = qa_chain.invoke({
            "input": query,
            "chat_history": formatted_history
        })

        answer = result.get("answer", "").strip()
        logger.info(f"Query: {query}\nAnswer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"An error occurred: {e}"