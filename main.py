import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils import fetch_pdf_data, split_text, initialize_vectorstore, initialize_retriever, initialize_qa_chain, ask_question
from dotenv import load_dotenv
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HRIS Chatbot API")
load_dotenv()

DOCUMENT_PATH = r"C:\Users\binay\Desktop\documentation_chatbot\epf_hris_docs.pdf"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

user_chat_histories = {}

file_path = DOCUMENT_PATH[8:] if DOCUMENT_PATH.startswith("file:///") else DOCUMENT_PATH


try:
    raw_text = fetch_pdf_data(file_path)
    if not raw_text:
        raise RuntimeError("Unable to fetch or extract text from the HRIS manual")
    texts = split_text(raw_text)
    vectorstore = initialize_vectorstore(texts, embedding_model="all-minilm")
    retriever = initialize_retriever(vectorstore)
    qa_chain = initialize_qa_chain(retriever, api_key=GEMINI_API_KEY)
    logger.info("HRIS Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize HRIS Chatbot: {str(e)}")
    raise RuntimeError(f"Initialization failed: {str(e)}")


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None 



class QueryResponse(BaseModel):
    query: str
    answer: str
    user_id: str
    response_time: float


@app.get("/")
def read_root():
    """Root endpoint for the HRIS Chatbot API."""
    return {"message": "Welcome to the HRIS Chatbot API"}


@app.post("/ask/", response_model=QueryResponse)
def ask_question_endpoint(request: QueryRequest):

    try:
        
        start_time = time.time()    

        
        if not request.user_id:
            user_id = str(uuid.uuid4())
            logger.info(f"Generated new user_id is: {user_id}")
        else:
            user_id = request.user_id
            logger.info(f"Using provided user_id which is: {user_id}")

        
        if user_id not in user_chat_histories:
            user_chat_histories[user_id] = []
            logger.info(f"Initialized chat history for user_id: {user_id}")

       
        answer = ask_question(qa_chain, request.query, chat_history=user_chat_histories[user_id])

        user_chat_histories[user_id].extend([f"User: {request.query}", f"Bot: {answer}"])

   
        if len(user_chat_histories[user_id]) > 20:
            user_chat_histories[user_id] = user_chat_histories[user_id][-20:]
            logger.info(f"chat history for user_id: {user_id}")

        response_time = time.time() - start_time

        logger.info(f"Processed query for user_id: {user_id}, response_time: {response_time:.2f}s")
        return {
            "query": request.query,
            "answer": answer,
            "user_id":user_id,
            "response_time": response_time

        }
    


    
    except Exception as e:
        logger.error(f"Error processing query for user_id {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Log startup event."""
    logger.info(f"HRIS Chatbot started with document: {DOCUMENT_PATH}")