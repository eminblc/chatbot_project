"""SRT subtitle file parsing utilities."""
import pysubs2
import json
from pathlib import Path
from src.models.scene import Scene
from config.constants import SCENE_GAP_THRESHOLD_SECONDS
from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_srt_into_scenes(srt_path: Path, gap_threshold: int = SCENE_GAP_THRESHOLD_SECONDS) -> list:
    """Split SRT file into scenes based on time gaps."""
    srt_path = Path(srt_path)
    
    subs = None
    for encoding in ['utf-8', 'cp1254', 'latin-1', 'iso-8859-1', 'windows-1252']:
        try:
            subs = pysubs2.load(str(srt_path), encoding=encoding)
            break
        except:
            continue
    
    if not subs:
        subs = pysubs2.load(str(srt_path))
    
    scenes = []
    current_scene_text = []
    start_time = subs[0].start

    for i, sub in enumerate(subs):
        current_scene_text.append(sub.text.replace("\\N", "\n").replace("\\n", "\n").strip())
        
        if i + 1 < len(subs):
            gap = (subs[i+1].start - sub.end) / 1000.0
            
            if gap > gap_threshold:
                scenes.append(Scene(
                    text=" ".join(current_scene_text),
                    start=start_time,
                    end=sub.end
                ))
                
                current_scene_text = []
                start_time = subs[i+1].start
    
    if current_scene_text:
        scenes.append(Scene(
            text=" ".join(current_scene_text),
            start=start_time,
            end=subs[-1].end
        ))
    
    return scenes


def save_srt_scenes_to_json(raw_dir: Path, processed_dir: Path) -> None:
    """Process SRT files in directory and save as JSON."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_srt(srt_path: Path):
        """Process single SRT file and save to JSON."""
        rel_path = srt_path.relative_to(raw_dir)
        json_path = (processed_dir / rel_path).with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_path.exists():
            return
        
        scenes = split_srt_into_scenes(srt_path)
        scenes_list = [s.to_dict(idx + 1) for idx, s in enumerate(scenes)]
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scenes_list, f, ensure_ascii=False, indent=4)
        
        logger.info("Processed %s: %d scenes", srt_path.name, len(scenes_list))
    
    srt_files = list(raw_dir.rglob("*.srt"))
    for srt_path in srt_files:
        process_single_srt(srt_path)
