from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from src.vector_store import embeddings, get_or_create_vector_db
from src.core.llm_engine import get_llm
from src.prompts.answer_prompt import prompt
from config.paths import get_series_paths
from config.constants import SERIES_FOLDER_NAME, RETRIEVAL_K, RETRIEVAL_SEARCH_TYPE, USE_LOCAL_LLM
from src.utils.logging import get_logger
import re

logger = get_logger(__name__)
_DIGIT_PATTERN = re.compile(r'\d+')

def build_rag_pipeline(series_name=None):
    """Load vector database for series."""
    target_series = series_name if series_name else SERIES_FOLDER_NAME
    logger.info("Building RAG for: %s", target_series)
    _, _, chroma_db_dir = get_series_paths(target_series)
    return get_or_create_vector_db(
        docs=[], 
        embedder=embeddings, 
        collection_name=target_series,
        persist_dir=chroma_db_dir
    )

def create_filtered_rag_chain(vector_store, filters=None, use_local=None):
    """Create RAG chain with optional metadata filtering."""
    search_kwargs = {"k": RETRIEVAL_K}
    
    if filters:
        filter_conditions = []
        if filters.get("season"):
            season_num = filters["season"]
            if isinstance(season_num, str):
                match = _DIGIT_PATTERN.search(season_num)
                season_num = int(match.group()) if match else int(season_num)
            filter_conditions.append({"season": {"$eq": season_num}})
        if filters.get("episode"):
            episode_num = filters["episode"]
            if isinstance(episode_num, str):
                match = _DIGIT_PATTERN.search(episode_num)
                episode_num = int(match.group()) if match else int(episode_num)
            filter_conditions.append({"episode_num": {"$eq": episode_num}})
        
        if filter_conditions:
            search_kwargs["filter"] = filter_conditions[0] if len(filter_conditions) == 1 else {"$and": filter_conditions}
            logger.info("Using filters: %s", search_kwargs["filter"])
    
    retriever = vector_store.as_retriever(
        search_type=RETRIEVAL_SEARCH_TYPE,
        search_kwargs=search_kwargs
    )
    
    is_local = use_local if use_local is not None else USE_LOCAL_LLM
    llm_instance = get_llm(is_local=is_local)
    
    doc_prompt = PromptTemplate.from_template(
        "--- SCENE ---\n"
        "SOURCE: {episode} | TIME: {start_time}\n"
        "CONTENT: {page_content}\n"
        "---------------"
    )
    question_answering_chain = create_stuff_documents_chain(
        llm=llm_instance, 
        prompt=prompt, 
        document_prompt=doc_prompt,
        document_separator="\n\n"
    )

    return create_retrieval_chain(retriever, question_answering_chain)
