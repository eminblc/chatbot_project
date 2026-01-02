"""Excel filtering utilities to remove dialogues from audio descriptions."""
import re
from typing import List, Dict
from config.constants import (
    MIN_ACTION_WORDS, 
    ACTION_INDICATORS, 
    DIALOGUE_QUESTION_STARTERS,
    DIALOGUE_VERBS,
    SPEAKER_PATTERNS
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def has_quotation_marks(text: str) -> bool:
    """Check if text contains any type of quotation marks."""
    return '"' in text or "'" in text or '"' in text or '"' in text or "'" in text or "'" in text


def is_question_dialogue(text: str) -> bool:
    """Check if text is a question that sounds like dialogue."""
    if text.strip().endswith('?'):
        text_lower = text.lower()
        for starter in DIALOGUE_QUESTION_STARTERS:
            if text_lower.startswith(starter):
                return True
    return False


def has_speaker_pattern(text: str) -> bool:
    """Check if text contains speaker patterns like 'Name:' or 'said/asked'."""
    for pattern in SPEAKER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def has_dialogue_verbs(text: str) -> bool:
    """Check if text contains dialogue-indicating verbs like 'said' or 'asked'."""
    text_lower = text.lower()
    for verb in DIALOGUE_VERBS:
        if re.search(rf'\b\w+\s+{verb}\b', text_lower):
            return True
    return False


def is_too_short(text: str) -> bool:
    """Check if text is too short to be a meaningful action description."""
    words = text.split()
    return len(words) < MIN_ACTION_WORDS


def has_action_indicators(text: str) -> bool:
    """Check if text contains multiple action-indicating words."""
    text_lower = text.lower()
    action_count = sum(1 for indicator in ACTION_INDICATORS if indicator in text_lower)
    return action_count >= 2


def is_dialogue_text(text: str) -> bool:
    """Return True if text is dialogue (filter), False if action (keep)."""
    if not text or not text.strip():
        return True
    
    text = text.strip()
    
    if has_quotation_marks(text):
        return True
    if is_question_dialogue(text):
        return True
    if has_speaker_pattern(text):
        return True
    if has_dialogue_verbs(text):
        return True
    if has_action_indicators(text):
        return False
    if is_too_short(text):
        return True
    
    return False


def filter_dialogues_from_actions(actions: List[Dict]) -> tuple[List[Dict], int, int]:
    """Filter dialogues from actions, return (filtered_actions, kept_count, removed_count)."""
    filtered = []
    removed_count = 0
    
    for action in actions:
        text = action.get('text', '')
        clean_text = text.replace('[ACTION: ', '').replace(']', '').strip()
        
        if is_dialogue_text(clean_text):
            removed_count += 1

        else:
            filtered.append(action)
    
    kept_count = len(filtered)
    
    logger.info("Dialogue filtering: kept %d actions, removed %d dialogues", 
                kept_count, removed_count)
    
    return filtered, kept_count, removed_count
