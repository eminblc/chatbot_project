"""Excel parsing utilities for action descriptions."""
import json
import pandas as pd
from pathlib import Path
from config.constants import ACTION_DURATION_MS
from src.utils.logging import get_logger
from src.preprocessing.excel_filter import filter_dialogues_from_actions

logger = get_logger(__name__)

def parse_time_to_ms(time_val) -> int:
    """Convert time formats (1:30, 90s) to milliseconds."""
    if pd.isna(time_val):
        return 0
    
    t_str = str(time_val).replace('s', '').strip()
    parts = t_str.split(':')
    
    total_seconds = 0
    if len(parts) == 3:
        total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        total_seconds = int(parts[0]) * 60 + float(parts[1])
    else:
        total_seconds = float(parts[0])
    
    return int(total_seconds * 1000)


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to 'HH:MM:SS,mmm' format."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"


def process_excel(file_path: Path, is_action: bool = False, filter_dialogues: bool = True) -> list:
    """
    Process scenes from Excel file.
    
    Args:
        file_path: Path to Excel file
        is_action: Whether to wrap text in [ACTION: ] tags
        filter_dialogues: Whether to filter out dialogue entries (default: True)
    
    Returns:
        List of scene dictionaries
    """
    df = pd.read_excel(file_path)
    results = []

    for _, row in df.iterrows():
        start_ms = parse_time_to_ms(row['Time'])
        end_ms = start_ms + ACTION_DURATION_MS
        
        text = str(row['Subtitle']).strip()
        
        if is_action:
            text = f"[ACTION: {text}]"
        
        results.append({
            "text": text,
            "start_time": ms_to_timestamp(start_ms),
            "end_time": ms_to_timestamp(end_ms),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "type": "action" if is_action else "dialogue"
        })
    
    # Filter dialogues if requested
    if filter_dialogues and is_action:
        results, kept, removed = filter_dialogues_from_actions(results)
        logger.info("Excel filtering for %s: kept %d, removed %d dialogues", 
                   file_path.name, kept, removed)
    
    return results


def save_excel_scenes_to_json(raw_dir: Path, processed_dir: Path, is_action: bool = False, 
                              filter_dialogues: bool = True) -> None:
    """
    Process Excel files in directory and save as JSON.
    
    Args:
        raw_dir: Directory containing raw Excel files
        processed_dir: Directory to save processed JSON files
        is_action: Whether to treat content as actions
        filter_dialogues: Whether to filter out dialogues (default: True for actions)
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_excel(excel_path: Path):
        """Process single Excel file and save to JSON."""
        rel_path = excel_path.relative_to(raw_dir)
        json_path = (processed_dir / rel_path).with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_path.exists():
            return
        
        scenes_data = process_excel(excel_path, is_action=is_action, 
                                    filter_dialogues=filter_dialogues)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, ensure_ascii=False, indent=4)
        
        logger.info("Processed %s: %d scenes", excel_path.name, len(scenes_data))
    
    excel_files = list(raw_dir.rglob("*.xlsx"))
    for excel_path in excel_files:
        process_single_excel(excel_path)
