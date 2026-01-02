"""Data processing module for creating vector databases from raw subtitle files."""
from src.preprocessing.srt_parser import save_srt_scenes_to_json
from src.preprocessing.excel_parser import save_excel_scenes_to_json
from src.vector_store import embeddings, text_splitter, get_or_create_vector_db
from config.paths import get_series_paths, get_series_subtitle_files_paths
from src.utils.data_loader import load_scenes_as_documents
from src.preprocessing.merger import merge_json_files
from src.utils.logging import get_logger
import shutil

logger = get_logger(__name__)


def process_series(series_name):
    """Process raw SRT and Excel files to create merged JSON files."""
    logger.info("Processing series: %s", series_name)
    
    _, _, chroma_db_dir = get_series_paths(series_name)
    raw_ad_files_path, raw_cs_files_path, proc_ad_files_path, proc_cs_files_path, proc_merged_path = get_series_subtitle_files_paths(series_name)

    # Process SRT files
    logger.info("Processing SRT files...")
    save_srt_scenes_to_json(raw_cs_files_path, proc_cs_files_path)
    
    # Process Excel files if they exist
    action_files_dict = {}
    if any(raw_ad_files_path.rglob("*.xlsx")):
        logger.info("Processing audio descriptions...")
        save_excel_scenes_to_json(raw_ad_files_path, proc_ad_files_path, is_action=True)
        
        # Build action files dictionary
        for action_file in proc_ad_files_path.rglob("*.json"):
            relative_path = action_file.relative_to(proc_ad_files_path)
            base_name = action_file.stem.replace("_audio_description", "")
            action_files_dict[(relative_path.parent, base_name)] = action_file
    
    # Merge or copy dialogue files
    logger.info("Creating merged files...")
    for dialogue_file in proc_cs_files_path.rglob("*.json"):
        relative_path = dialogue_file.relative_to(proc_cs_files_path)
        output_file = proc_merged_path / relative_path.parent / (dialogue_file.stem + "_merged.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        action_file = action_files_dict.get((relative_path.parent, dialogue_file.stem))
        if action_file:
            merge_json_files(str(dialogue_file), str(action_file), str(output_file))
        else:
            shutil.copy(str(dialogue_file), str(output_file))

    logger.info("Loading merged data...")
    clean_data = load_scenes_as_documents(proc_merged_path, series_name)

    logger.info("Splitting documents into chunks...")
    docs = []
    if clean_data:
        docs = text_splitter.split_documents(clean_data)
    logger.info("Created %d document chunks", len(docs))

    logger.info("Creating/updating vector database...")
    get_or_create_vector_db(
        docs=docs, 
        embedder=embeddings, 
        collection_name=series_name,
        persist_dir=chroma_db_dir
    )
    
    logger.info("Processing complete!")
