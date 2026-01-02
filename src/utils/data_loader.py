"""Data loading utilities."""
import json
import os
import re
from langchain.schema import Document
from src.utils.logging import get_logger

logger = get_logger(__name__)

def extract_season_episode(filename):
    """Extract season/episode from filename (e.g., s1_e3 -> 1,3)."""
    match = re.search(r's(\d+)_e(\d+)', filename.lower())
    if match:
        return int(match.group(1)), int(match.group(2))
    logger.warning("No season/episode in: %s", filename)
    return None, None


def load_scenes_as_documents(processed_dir, series_folder_name):
    """Load scenes from JSON files as LangChain Documents."""
    clean_data = []
    json_files = sorted(list(processed_dir.rglob("*.json")))
    
    if not json_files:
        logger.warning("No JSON files in %s", processed_dir)
        return clean_data
    
    logger.info("Loading %d JSON files...", len(json_files))
    
    for j_path in json_files:
        try:
            season, episode = extract_season_episode(j_path.stem)
            with open(j_path, "r", encoding="utf-8") as f:
                scenes_data = json.load(f)
                for s in scenes_data:
                    metadata = {
                        "source": j_path.name,
                        "episode": j_path.stem,
                        "season": season,
                        "episode_num": episode,
                        "series": series_folder_name,
                        "start_time": s["start_time"],
                        "end_time": s["end_time"],
                        "scene_id": s["scene_id"]
                    }
                    clean_data.append(Document(page_content=s["text"], metadata=metadata))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed: %s - %s", j_path.name, e)
    
    logger.info("Loaded %d scenes from %d files", len(clean_data), len(json_files))
    return clean_data
