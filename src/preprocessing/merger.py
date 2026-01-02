import json
import os
from typing import Dict, List, Tuple
from config.constants import NGRAM_SIZE, TIME_WINDOW_MS
from src.utils.logging import get_logger
from src.utils.text_processing import normalize_text, build_ngrams

logger = get_logger(__name__)


def _build_time_buckets(dialogues: List[Dict]) -> Dict[int, List[Dict]]:
    """Group dialogues by time windows for efficient filtering."""
    dialogue_by_time = {}
    for d in dialogues:
        start_ms = d.get('start_ms', 0)
        time_bucket = start_ms // TIME_WINDOW_MS
        if time_bucket not in dialogue_by_time:
            dialogue_by_time[time_bucket] = []
        dialogue_by_time[time_bucket].append(d)
    return dialogue_by_time


def _find_potential_overlaps(action: Dict, dialogue_by_time: Dict[int, List[Dict]]) -> List[Dict]:
    """Find dialogues that overlap with action in time."""
    act_start = action.get('start_ms', 0)
    act_end = action.get('end_ms', 0)
    time_bucket = act_start // TIME_WINDOW_MS
    potential_overlaps = []
    
    for bucket_offset in range(-120, 121):
        bucket = time_bucket + bucket_offset
        if bucket in dialogue_by_time:
            for d in dialogue_by_time[bucket]:
                d_start = d.get('start_ms', 0)
                d_end = d.get('end_ms', 0)
                
                if d_start <= act_start and act_end <= d_end:
                    potential_overlaps.append(d)
                elif act_start <= d_end and d_start <= act_end:
                    potential_overlaps.append(d)
    
    return potential_overlaps


def _is_duplicate_action(action_text: str, potential_overlaps: List[Dict]) -> bool:
    """Check if action is duplicate using substring, word sequence, and n-gram matching."""
    if not potential_overlaps:
        return False
    
    act_words = action_text.split()
    
    if len(act_words) < 5:
        for po in potential_overlaps:
            po_text = normalize_text(po.get('text', ''))
            if action_text == po_text:
                return True
        return False
    
    for po in potential_overlaps:
        po_text = normalize_text(po.get('text', ''))
        if not po_text:
            continue
        
        if action_text in po_text:
            return True
        
        dia_words = po_text.split()
        if _check_word_sequence_match(act_words, dia_words):
            return True
        
        act_ngrams = build_ngrams(action_text, NGRAM_SIZE)
        dia_ngrams = build_ngrams(po_text, NGRAM_SIZE)
        
        if act_ngrams and dia_ngrams:
            overlap = len(act_ngrams & dia_ngrams)
            overlap_ratio = overlap / len(act_ngrams)
            
            if overlap_ratio > 0.8:
                return True
    
    return False


def _check_word_sequence_match(action_words: List[str], dialogue_words: List[str]) -> bool:
    """Check if 80%+ of action words appear in dialogue in the same order."""
    if not action_words or not dialogue_words:
        return False
    
    matched_count = 0
    dia_idx = 0
    
    for act_word in action_words:
        while dia_idx < len(dialogue_words):
            if dialogue_words[dia_idx] == act_word:
                matched_count += 1
                dia_idx += 1
                break
            dia_idx += 1
    
    match_ratio = matched_count / len(action_words)
    return match_ratio >= 0.8


def _merge_and_deduplicate(dialogues: List[Dict], actions: List[Dict], 
                           dialogue_by_time: Dict) -> Tuple[List[Dict], int, int]:
    """Merge actions into dialogues with duplicate detection."""
    merged_list = list(dialogues)
    added_actions = 0
    skipped_duplicates = 0
    
    for act in actions:
        raw_text = act.get('text', '')
        clean_text = raw_text.replace('[ACTION: ', '').replace(']', '').strip()
        act_text_clean = normalize_text(clean_text)
        
        if not act_text_clean:
            continue
        
        if len(act_text_clean) < 3:
            act['type'] = "action"
            merged_list.append(act)
            added_actions += 1
            continue
        
        potential_overlaps = _find_potential_overlaps(act, dialogue_by_time)
        is_duplicate = _is_duplicate_action(act_text_clean, potential_overlaps)
        
        if is_duplicate:
            skipped_duplicates += 1
        else:
            act['type'] = "action"
            merged_list.append(act)
            added_actions += 1
    
    return merged_list, added_actions, skipped_duplicates


def _sort_chronologically(merged_list: List[Dict]) -> None:
    """Sort scenes by start time."""
    merged_list.sort(key=lambda x: x.get('start_ms', 0))


def _renumber_scene_ids(merged_list: List[Dict]) -> None:
    """Renumber scene IDs sequentially."""
    for i, item in enumerate(merged_list):
        item['scene_id'] = i + 1


def _save_merged_file(merged_list: List[Dict], output_path: str) -> None:
    """Save merged scenes to JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, indent=4, ensure_ascii=False)


def merge_json_files(dialogue_path: str, action_path: str, output_path: str) -> None:
    """Merge dialogue and action JSONs with duplicate detection."""
    logger.info("Loading files...")
    
    try:
        with open(dialogue_path, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
            
        with open(action_path, 'r', encoding='utf-8') as f:
            actions = json.load(f)
            
        logger.info("Dialogues: %d lines, Actions: %d lines", len(dialogues), len(actions))
        
        dialogue_by_time = _build_time_buckets(dialogues)
        
        merged_list, added_actions, skipped_duplicates = _merge_and_deduplicate(
            dialogues, actions, dialogue_by_time
        )
        
        _sort_chronologically(merged_list)
        _renumber_scene_ids(merged_list)
        _save_merged_file(merged_list, output_path)
        
        logger.info("Merge successful: %d actions added, %d duplicates removed, %d total lines", 
                   added_actions, skipped_duplicates, len(merged_list))
        
    except FileNotFoundError as e:
        logger.error("File not found: %s", e.filename)
    except (ValueError, OSError, IOError) as e:
        logger.error("Error in merge: %s", e)
