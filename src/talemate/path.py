import os
from pathlib import Path

__all__ = [
    "TALEMATE_ROOT",
    "SCENES_DIR",
    "TEMPLATES_DIR",
    "TTS_DIR",
    "LOGS_DIR",
    "CONFIG_FILE",
    "relative_to_root",
]

# Allow data (config/scenes/templates/tts/logs) to live apart from the
# source tree: set TALEMATE_ROOT to a data directory. Defaults to the
# repository root for unchanged behavior.
TALEMATE_ROOT = Path(os.environ["TALEMATE_ROOT"]) if os.environ.get("TALEMATE_ROOT") else Path(__file__).parent.parent.parent
SCENES_DIR = TALEMATE_ROOT / "scenes"
TEMPLATES_DIR = TALEMATE_ROOT / "templates"
TTS_DIR = TALEMATE_ROOT / "tts"
LOGS_DIR = TALEMATE_ROOT / "logs"


CONFIG_FILE = TALEMATE_ROOT / "config.yaml"


def relative_to_root(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path.resolve().relative_to(TALEMATE_ROOT)
