
"""Main CLI for processing subtitles."""
import argparse
from src.core.data_processor import process_series
from src.utils.logging import setup_logging, get_logger


setup_logging()
logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TV Series RAG - Process subtitles and create vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--process',
        action='store_true',
        help='Process raw subtitle files and create vector database'
    )
    parser.add_argument(
        '--series',
        type=str,
        default='stranger_things',
        help='Series name to process (default: stranger_things)'
    )
    
    args = parser.parse_args()
    
    if not args.process:
        parser.error("Must specify --process flag")
    
    try:
        process_series(args.series)
    except (ValueError, FileNotFoundError, OSError) as e:
        logger.error("Error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
