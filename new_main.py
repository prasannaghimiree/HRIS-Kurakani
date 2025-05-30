import time
import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from new_utils import fetch_pdf_data, split_text, initialize_vectorstore, initialize_retriever, initialize_qa_chain, ask_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HRIS Chatbot API")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DOCUMENT_PATH = "file:///C:/Users/binay/Desktop/documentation_chatbot/Tutorial.pdf"
file_path = DOCUMENT_PATH[8:] if DOCUMENT_PATH.startswith("file:///") else DOCUMENT_PATH

user_chat_histories = {}
templates = Jinja2Templates(directory="templates")

# Initialize chatbot
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
    return {"message": "Welcome to the HRIS Chatbot API"}

@app.post("/ask/", response_model=QueryResponse)
async def ask_question_endpoint(request: QueryRequest):
    """Handle API query with user-specific chat history."""
    try:
        start_time = time.time()

        user_id = request.user_id or str(uuid.uuid4())
        logger.info(f"{'Generated new' if not request.user_id else 'Using provided'} user_id: {user_id}")

        if user_id not in user_chat_histories:
            user_chat_histories[user_id] = []
            logger.info(f"Initialized chat history for user_id: {user_id}")

        answer = ask_question(qa_chain, request.query, chat_history=user_chat_histories[user_id])
        user_chat_histories[user_id].extend([f"User: {request.query}", f"Bot: {answer}"])

        if len(user_chat_histories[user_id]) > 20:
            user_chat_histories[user_id] = user_chat_histories[user_id][-20:]

        logger.debug(f"Updated chat history for user_id {user_id}: {user_chat_histories[user_id]}")
        response_time = time.time() - start_time
        logger.info(f"Processed query for user_id: {user_id}, response_time: {response_time:.2f}s")

        return {
            "query": request.query,
            "answer": answer,
            "user_id": user_id,
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Error processing query for user_id {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Render chat page."""
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "query": "",
        "answer": ""
    })

@app.post("/chat", response_class=HTMLResponse)
async def post_chat_page(request: Request, query: str = Form(...)):
    """Handle web form submission."""
    try:
        user_id = "default_user"
        if user_id not in user_chat_histories:
            user_chat_histories[user_id] = []

        answer = ask_question(qa_chain, query, chat_history=user_chat_histories[user_id])
        user_chat_histories[user_id].extend([f"User: {query}", f"Bot: {answer}"])

        if len(user_chat_histories[user_id]) > 20:
            user_chat_histories[user_id] = user_chat_histories[user_id][-20:]

        logger.debug(f"Updated chat history for user_id {user_id}: {user_chat_histories[user_id]}")
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "query": query,
            "answer": answer
        })
    except Exception as e:
        logger.error(f"Error in web form processing: {str(e)}")
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "query": query,
            "answer": f"Error: {str(e)}"
        })

@app.on_event("startup")
async def startup_event():
    logger.info(f"HRIS Chatbot started with document: {DOCUMENT_PATH}")