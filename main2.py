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

from utils import fetch_pdf_data,split_text,initialize_vectorstore,initialize_retriever,initialize_qa_chain,ask_question


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HRIS Chatbot API")

from langsmith import utils
utils.tracing_is_enabled()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


DOCUMENT_PATH = r"C:\Users\binay\Desktop\documentation_chatbot\epf_hris_docs.pdf"
file_path = DOCUMENT_PATH[8:] if DOCUMENT_PATH.startswith("file:///") else DOCUMENT_PATH


user_chat_histories = {}

templates = Jinja2Templates(directory="templates")


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
def ask_question_endpoint(request: QueryRequest):
    try:
        start_time = time.time()

        if not request.user_id:
            user_id = str(uuid.uuid4())
            logger.info(f"Generated new user_id: {user_id}")
        else:
            user_id = request.user_id
            logger.info(f"Using provided user_id: {user_id}")

        if user_id not in user_chat_histories:
            user_chat_histories[user_id] = []
            logger.info(f"Initialized chat history for user_id: {user_id}")

        answer = ask_question(qa_chain, request.query, chat_history=user_chat_histories[user_id])
        user_chat_histories[user_id].extend([f"User: {request.query}", f"Bot: {answer}"])

        if len(user_chat_histories[user_id]) > 20:
            user_chat_histories[user_id] = user_chat_histories[user_id][-20:]

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
def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "query": "",
        "answer": ""
    })

@app.post("/chat", response_class=HTMLResponse)
async def post_chat_page(request: Request, query: str = Form(...)):
    try:
        user_id = "default_user"

        if user_id not in user_chat_histories:
            user_chat_histories[user_id] = []

        answer = ask_question(qa_chain, query, chat_history=user_chat_histories[user_id])
        user_chat_histories[user_id].extend([f"User: {query}", f"Bot: {answer}"])

        if len(user_chat_histories[user_id]) > 20:
            user_chat_histories[user_id] = user_chat_histories[user_id][-20:]

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

# Startup log
@app.on_event("startup")
async def startup_event():
    logger.info(f"HRIS Chatbot started with document: {DOCUMENT_PATH}")
