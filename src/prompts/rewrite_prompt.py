"""Query rewrite and optimization prompt for RAG system."""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.llm_engine import llm
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_INSTRUCTIONS = (
    "You are an Advanced Query Optimizer for a TV Series Subtitle Search System.\n"
    "Database: SRT subtitle chunks with metadata (season, episode, timestamps).\n"
    "Goal: Transform user questions into optimal search queries that maximize retrieval accuracy.\n\n"
    
    "### OUTPUT FORMAT (STRICT)\n"
    "Return ONLY valid JSON. No explanations. No extra keys.\n"
    "Follow the schema in FORMAT INSTRUCTIONS exactly.\n\n"
)

_TRANSLATION_RULES = (
    "## 1) real_question\n"
    "   - Translate the user's question to clear, natural ENGLISH\n"
    "   - Preserve the question format\n"
    "   - Examples:\n"
    "     * 'Will neden kayboldu?' → 'Why did Will disappear?'\n"
    "     * 'Birinci sezonda kim öldü?' → 'Who died in season one?'\n"
    "     * 'Hopper yaşıyor mu?' → 'Is Hopper alive?'\n\n"
)

_SEARCH_EXPANSION_RULES = (
    "## 2) search_terms (30-50 terms)\n"
    "   Generate comprehensive search keywords using these categories:\n\n"
    
    "   A) DIRECT ENTITIES: Character names, nicknames, aliases (Turkish + English)\n"
    "   B) CONTEXTUAL ROLES: Occupations, relationships, groups\n"
    "   C) LOCATIONS: Generic (lab, school) + Specific (Hawkins Lab, Starcourt)\n"
    "   D) ACTIONS & EVENTS: Verbs, states, events\n"
    "   E) OBJECTS & PROPS: Key items, documents\n"
    "   F) EMOTIONAL & DIALOGUE CUES: Intent, emotion\n"
    "   G) SCENE TYPES: finale, opening, flashback, etc.\n\n"
    
    "   SMART EXPANSION:\n"
    "   - Death/Status: Add 'body', 'funeral', 'alive', 'survived', 'killed', 'hospital'\n"
    "   - Character Arcs: Include recurring motifs (Hopper→'Russia','prison')\n"
    "   - Mystery: Add 'because', 'why', 'how', 'explain', 'reason'\n"
    "   - Vague pronouns: Aggressively expand with possible referents\n\n"
)

_FILTER_EXTRACTION_RULES = (
    "## 3) filters (CRITICAL)\n"
    "   Extract season/episode from these patterns:\n\n"
    
    "   TURKISH: 'birinci sezon'→\"1\", 'ikinci sezon'→\"2\", 'beşinci bölüm'→episode:\"5\"\n"
    "   ENGLISH: 'season 1'→\"1\", 'episode 3'→\"3\", 's2e5'→season:\"2\",episode:\"5\"\n\n"
    
    "   RULES:\n"
    "   - No mention → season:\"\", episode:\"\"\n"
    "   - Convert words to digits: 'birinci'→\"1\", 'second'→\"2\"\n"
    "   - Finale without episode → estimate typical length\n\n"
)

_SERIES_DETECTION_RULES = (
    "## 4) detected_series\n"
    "   Auto-detect TV series from character names/locations:\n\n"
    
    "   CHARACTER SIGNATURES:\n"
    "   - Stranger Things: Will, Eleven, Mike, Dustin, Joyce, Hopper, Demogorgon, Hawkins, Vecna\n"
    "   - Breaking Bad: Walter White, Jesse Pinkman, Hank, Saul, Gus Fring, Heisenberg, DEA\n\n"
    
    "   VALID VALUES:\n"
    "   - \"stranger_things\" - for Stranger Things\n"
    "   - \"breaking_bad\" - for Breaking Bad\n"
    "   - \"\" - unable to determine or generic\n\n"
)

_EXAMPLES = (
    "### EXAMPLES\n\n"
    
    "Example 1: 'Birinci sezonun sonunda Will'e ne oldu?'\n"
    "{{\n"
    "  \"real_question\": \"What happened to Will at the end of season one?\",\n"
    "  \"search_terms\": [\"Will\", \"Will Byers\", \"season finale\", \"Upside Down\", \"rescue\", \"hospital\", \"Joyce\", \"Hopper\"],\n"
    "  \"filters\": {{\"season\": \"1\", \"episode\": \"\"}},\n"
    "  \"detected_series\": \"stranger_things\"\n"
    "}}\n\n"
    
    "Example 2: 'Hopper öldü mü?'\n"
    "{{\n"
    "  \"real_question\": \"Did Hopper die?\",\n"
    "  \"search_terms\": [\"Hopper\", \"Jim Hopper\", \"dead\", \"alive\", \"death\", \"explosion\", \"sacrifice\", \"Russia\"],\n"
    "  \"filters\": {{\"season\": \"\", \"episode\": \"\"}},\n"
    "  \"detected_series\": \"stranger_things\"\n"
    "}}\n\n"
    
    "Example 3: 'Walter kim?'\n"
    "{{\n"
    "  \"real_question\": \"Who is Walter?\",\n"
    "  \"search_terms\": [\"Walter\", \"Walter White\", \"Heisenberg\", \"teacher\", \"chemistry\", \"cancer\"],\n"
    "  \"filters\": {{\"season\": \"\", \"episode\": \"\"}},\n"
    "  \"detected_series\": \"breaking_bad\"\n"
    "}}\n\n"
)


schema = {
    "type": "object",
    "properties": {
        "real_question": {"type": "string"},
        "search_terms": {
            "type": "array",
            "items": {"type": "string"}
        },
        "filters": {
            "type": "object",
            "properties": {
                "season": {"type": "string"},
                "episode": {"type": "string"}
            },
            "required": ["season", "episode"]
        },
        "detected_series": {"type": "string"}
    },
    "required": ["real_question", "search_terms", "filters", "detected_series"]
}
parser = JsonOutputParser(schema=schema)

REWRITE_PROMPT = PromptTemplate(
    template=(
        _SYSTEM_INSTRUCTIONS +
        "### TASK BREAKDOWN\n\n" +
        _TRANSLATION_RULES +
        _SEARCH_EXPANSION_RULES +
        _FILTER_EXTRACTION_RULES +
        _SERIES_DETECTION_RULES +
        _EXAMPLES +
        "### FORMAT INSTRUCTIONS\n"
        "{format_instructions}\n\n"
        "### USER QUESTION\n"
        "{question}\n\n"
        "Remember: Output ONLY valid JSON. No markdown, no explanations.\n"
    ),
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

rewriter_chain = REWRITE_PROMPT | llm | parser


def optimized_rag_ask(user_query: str) -> tuple:
    """Optimize user query for better retrieval."""
    try:
        result = rewriter_chain.invoke({"question": user_query})
        real_q = result.get("real_question", "")
        terms = result.get("search_terms", [])
        filters = result.get("filters", {})
        detected_series = result.get("detected_series", "")
        
        season_filter = filters.get("season", "")
        episode_filter = filters.get("episode", "")
        
        logger.info("Query: %s → %s | Series: %s | Season: %s | Episode: %s | Terms: %d", 
                   user_query[:50], real_q[:50], detected_series or "?",
                   season_filter or "?", episode_filter or "?", len(terms))

        combined_query = f"{real_q} | TERMS: {', '.join(terms)}"
        
        return (combined_query, filters, detected_series)

    except (ValueError, KeyError, TypeError) as e:
        logger.error("Query rewrite failed: %s", e)
        return (user_query, {}, "")
