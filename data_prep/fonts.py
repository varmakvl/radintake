"""Fetch the handwriting fonts used for handwritten fields into data/fonts/ (gitignored).

    python -m data_prep.fonts

All fonts come from the google/fonts repository under open licences (SIL OFL 1.1 or Apache 2.0).
They are fetched once and pinned by SHA-256 in data/fonts/FONTS.json so renders are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/google/fonts/main/"
FONTS = {
    # family -> repo path (licence)
    "Caveat": "ofl/caveat/Caveat[wght].ttf",
    "HomemadeApple": "apache/homemadeapple/HomemadeApple-Regular.ttf",
    "ReenieBeanie": "ofl/reeniebeanie/ReenieBeanie.ttf",
    "NothingYouCouldDo": "ofl/nothingyoucoulddo/NothingYouCouldDo.ttf",
}
FONT_DIR = Path("data/fonts")


def font_files(font_dir: Path = FONT_DIR) -> dict[str, Path]:
    """Family -> local TTF path for every font that is present."""
    return {fam: font_dir / Path(rel).name for fam, rel in FONTS.items() if (font_dir / Path(rel).name).is_file()}


def fetch(font_dir: Path = FONT_DIR) -> dict[str, str]:
    font_dir.mkdir(parents=True, exist_ok=True)
    pins: dict[str, str] = {}
    for fam, rel in FONTS.items():
        dest = font_dir / Path(rel).name
        if not dest.is_file():
            url = BASE + rel.replace("[", "%5B").replace("]", "%5D")
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if data[:4] not in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
                raise RuntimeError(f"{fam}: {url} did not return a font file")
            dest.write_bytes(data)
        pins[fam] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (font_dir / "FONTS.json").write_text(json.dumps({"source": BASE, "files": FONTS, "sha256": pins}, indent=2),
                                         encoding="utf-8")
    return pins


if __name__ == "__main__":
    for fam, h in fetch().items():
        print(f"{fam}: {h[:16]}")
    sys.exit(0)
