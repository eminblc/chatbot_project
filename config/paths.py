""""Project Path Configurations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
CHROMA_DB = DATA / "chroma_db"

def get_series_paths(series_name):
    """Return raw, processed, and ChromaDB paths for series."""
    raw_dir = DATA_RAW / series_name
    processed_dir = DATA_PROCESSED / series_name
    chroma_db_dir = CHROMA_DB / series_name
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
    if not chroma_db_dir.exists():
        chroma_db_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, chroma_db_dir

def get_series_subtitle_files_paths(series_name):
    """Return subtitle and audio description paths for series."""
    raw_dir, processed_dir, _ = get_series_paths(series_name)
    raw_ad_files_path = raw_dir / "audio_descriptions"
    raw_cs_files_path = raw_dir / "captioned_subtitles"
    if not raw_ad_files_path.exists():
        raw_ad_files_path.mkdir(parents=True, exist_ok=True)
    if not raw_cs_files_path.exists():
        raw_cs_files_path.mkdir(parents=True, exist_ok=True)
    proc_ad_files_path = processed_dir / "audio_descriptions"
    proc_cs_files_path = processed_dir / "captioned_subtitles"
    merged_dir = processed_dir / "merged"
    if not merged_dir.exists():
        merged_dir.mkdir(parents=True, exist_ok=True)
    return raw_ad_files_path, raw_cs_files_path, proc_ad_files_path, proc_cs_files_path, merged_dir
