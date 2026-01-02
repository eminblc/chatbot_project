"""Central configuration constants for the chatbot system."""

SERIES_FOLDER_NAME = "stranger_things"
COLLECTION_NAME = SERIES_FOLDER_NAME

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

RETRIEVAL_K = 5
RETRIEVAL_SEARCH_TYPE = "similarity"

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 8000
USE_LOCAL_LLM = False

LOCAL_MODEL_NAME = "qwen2.5:7b"
GOOGLE_MODEL_NAME = "gemini-3-flash-preview"

SCENE_GAP_THRESHOLD_SECONDS = 12
ACTION_DURATION_MS = 3000

NGRAM_SIZE = 3
TIME_WINDOW_MS = 1000

# Embedding Configuration
EMBEDDING_MODEL = "models/text-embedding-004"

# Excel Filtering Constants
MIN_ACTION_WORDS = 5

DIALOGUE_QUESTION_STARTERS = [
    'what if', 'where', 'when', 'why', 'how', 'who', 'whose',
    'can you', 'will you', 'do you', 'did you', 'have you',
    'are you', 'is he', 'is she', 'is it', 'was', 'were'
]

DIALOGUE_VERBS = [
    'said', 'asked', 'replied', 'answered', 'whispered', 
    'shouted', 'yelled', 'muttered', 'explained', 'told'
]

SPEAKER_PATTERNS = [
    r'^[A-Z][a-z]+\s*:',
    r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*:',
    r',\s*[A-Z][a-z]+\s*:',
    r'\b(he|she|they|mom|dad|mother|father)\s+said\b',
    r'\b(he|she|they|mom|dad)\s+(asked|replied|whispered|shouted|yelled)\b'
]

DIALOGUE_INDICATORS = [
    '"', "'", ':', 'said', 'asked', 'replied', 'whispered', 'shouted', 
    'yelled', 'muttered', 'explained', 'told', 'answered'
]

ACTION_INDICATORS = [
    'appears', 'walks', 'looks', 'smiles', 'enters', 'exits', 'sits',
    'stands', 'opens', 'closes', 'picks', 'puts', 'takes', 'gives',
    'turns', 'moves', 'runs', 'jumps', 'falls', 'rises', 'touches',
    'holds', 'stares', 'watches', 'listens', 'nods', 'shakes'
]