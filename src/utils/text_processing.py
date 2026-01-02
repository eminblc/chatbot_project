"""Text processing utilities."""
import re
from nltk.util import ngrams as create_ngrams
from config.constants import NGRAM_SIZE

_BRACKET_PATTERN = re.compile(r'\[.*?\]')
_NON_WORD_PATTERN = re.compile(r'[^\w\s]')

def normalize_text(text):
    """Normalize text: remove brackets/special chars, lowercase."""
    text = _BRACKET_PATTERN.sub(' ', str(text))
    text = _NON_WORD_PATTERN.sub('', text.lower())
    return ' '.join(text.split())

def build_ngrams(text, n=NGRAM_SIZE):
    """Build n-gram set for text matching."""
    words = text.split()
    return {text} if len(words) < n else {" ".join(gram) for gram in create_ngrams(words, n)}
