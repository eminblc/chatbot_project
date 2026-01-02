"""Vector Store Utilities."""
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from config.constants import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

logger.info("Using Google Embedding: %s", EMBEDDING_MODEL)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

def get_or_create_vector_db(docs, embedder, collection_name, persist_dir):
    """Create or load Chroma vector store."""
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info("Loading existing database: %s", collection_name)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_dir
        )
    else:
        logger.info("Creating database '%s' with %d docs", collection_name, len(docs))
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedder,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        logger.info("Database created: %s", persist_dir)
    return vector_store