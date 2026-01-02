"""Multi-series query service."""
from typing import Dict, List, Optional
from src.core.pipeline import build_rag_pipeline, create_filtered_rag_chain
from src.prompts.rewrite_prompt import optimized_rag_ask
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SeriesQueryResult:
    """Query result from a single series."""
    def __init__(self, series_name: str, answer: str, sources: List[Dict], optimized_query: str):
        self.series_name = series_name
        self.answer = answer
        self.sources = sources
        self.optimized_query = optimized_query


class MultiSeriesService:
    """Handle multi-series queries with auto-detection."""
    AVAILABLE_SERIES = ["stranger_things", "breaking_bad"]
    
    def __init__(self):
        self.logger = logger
    
    def detect_target_series(self, query: str) -> Optional[str]:
        """Detect which series the query is about."""
        try:
            _, _, detected_series = optimized_rag_ask(query)
            if detected_series and detected_series in self.AVAILABLE_SERIES:
                self.logger.info("Auto-detected: %s", detected_series)
                return detected_series
        except (ValueError, KeyError, TypeError) as e:
            self.logger.warning("Detection failed: %s", e)
        return None
    
    def query_single_series(self, series_name: str, query: str, 
                           season: Optional[int] = None,
                           episode: Optional[int] = None,
                           use_local: Optional[bool] = None) -> SeriesQueryResult:
        """Query single series and return results."""
        vector_store = build_rag_pipeline(series_name=series_name)
        
        try:
            optimized_query, filters, _ = optimized_rag_ask(query)
            if season:
                filters['season'] = str(season)
            if episode:
                filters['episode'] = str(episode)
            self.logger.info("Filters: %s", filters)
        except (ValueError, KeyError) as e:
            self.logger.warning("Query optimization failed, using original: %s", e)
            optimized_query = query
            filters = {}
        
        rag_chain = create_filtered_rag_chain(vector_store, filters, use_local=use_local)
        response = rag_chain.invoke({"input": optimized_query})
        sources = self._format_sources(response["context"], series_name)
        
        return SeriesQueryResult(
            series_name=series_name,
            answer=response["answer"],
            sources=sources,
            optimized_query=optimized_query
        )
    
    def query_all_series(self, query: str, season: Optional[int] = None,
                        episode: Optional[int] = None,
                        use_local: Optional[bool] = None) -> Dict:
        """Query all series or auto-detected series and merge results."""
        detected_series = self.detect_target_series(query)
        
        if detected_series:
            result = self.query_single_series(detected_series, query, season, episode, use_local)
            return self._format_single_series_response(result, auto_detected=True)
        
        self.logger.info("Querying all: %s", ", ".join(self.AVAILABLE_SERIES))
        
        all_results = []
        for series_name in self.AVAILABLE_SERIES:
            try:
                result = self.query_single_series(series_name, query, season, episode, use_local)
                all_results.append(result)
            except (ValueError, FileNotFoundError, OSError) as e:
                self.logger.error("Error querying %s: %s", series_name, e)
        
        return self._merge_series_results(query, all_results)
    
    def _format_sources(self, context_docs: List, series_name: str) -> List[Dict]:
        """Format source documents to structured dicts."""
        sources = []
        for doc in context_docs:
            sources.append({
                "season": doc.metadata.get("season", "Unknown"),
                "episode": doc.metadata.get("episode", "Unknown"),
                "episode_num": doc.metadata.get("episode_num", "Unknown"),
                "series": series_name,
                "time": doc.metadata.get("start_time", "00:00"),
                "content": doc.page_content[:100] + "..."
            })
        return sources
    
    def _format_single_series_response(self, result: SeriesQueryResult, auto_detected: bool = False) -> Dict:
        """Format single series result to API response."""
        return {
            "status": "success",
            "original_query": result.optimized_query,
            "optimized_query": result.optimized_query,
            "answer": result.answer,
            "sources": result.sources,
            "source_count": len(result.sources),
            "series_queried": [result.series_name],
            "auto_detected": auto_detected
        }
    
    def _merge_series_results(self, original_query: str, results: List[SeriesQueryResult]) -> Dict:
        """Merge results from multiple series."""
        all_sources = []
        answers = []
        series_queried = []
        
        for result in results:
            for source in result.sources:
                source["series"] = result.series_name
            all_sources.extend(result.sources)
            answers.append(f"[{result.series_name.upper()}]: {result.answer}")
            series_queried.append(result.series_name)
        
        return {
            "status": "success",
            "original_query": original_query,
            "optimized_query": results[0].optimized_query if results else original_query,
            "answer": "\n\n".join(answers),
            "sources": all_sources,
            "source_count": len(all_sources),
            "series_queried": series_queried,
            "auto_detected": False
        }
