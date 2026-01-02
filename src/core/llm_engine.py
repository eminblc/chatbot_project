"""LLM Engine using Google Generative AI."""
from langchain_ollama import OllamaLLM
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from config.constants import (
    LOCAL_MODEL_NAME, 
    GOOGLE_MODEL_NAME,
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS, 
    USE_LOCAL_LLM
)
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((ValueError, OSError, ConnectionError)),
    before_sleep=lambda retry_state: logger.warning(
        "LLM init attempt %d failed, retrying...", retry_state.attempt_number
    )
)
def get_llm(is_local=USE_LOCAL_LLM):
    """Get LLM instance (local Ollama or Google API) with retry logic."""
    if is_local:
        logger.info("Initializing local LLM: %s", LOCAL_MODEL_NAME)
        return OllamaLLM(model=LOCAL_MODEL_NAME, temperature=LLM_TEMPERATURE)
    else:
        logger.info("Initializing Google LLM: %s", GOOGLE_MODEL_NAME)
        return GoogleGenerativeAI(
            model=GOOGLE_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

llm = get_llm()
