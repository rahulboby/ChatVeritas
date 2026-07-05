from pathlib import Path
import re

import trafilatura

"""
    Scrape URLs and save the content as text files in OUTPUT_DIR. The text files are named based on the URL, sanitized to be valid filenames.
"""

# ==============================================================================
# EDIT THESE
# ==============================================================================

URLS = [
    "https://en.wikipedia.org/wiki/Ford_Mustang",
]

# Saved relative to Path(__file__).parent.parent
OUTPUT_DIR = "data/raw/"

# ==============================================================================


def sanitize_filename(name: str) -> str:
    """Convert a string into a valid filename."""
    name = re.sub(r"https?://", "", name)
    name = re.sub(r"[^\w\-_.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:150]

def scrape_url(url: str, output_dir: Path):

    #First convert to markdown, then save to a text file. This preserves paragraphs and formatting, which is important for the next step of converting to a PDF.
    print(f"Scraping: {url}")

    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        print("  Failed to download.")
        return

    text = trafilatura.extract(
        downloaded,
        output_format="markdown",   # <-- Preserve paragraphs & formatting
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )

    if not text:
        print("  No content extracted.")
        return

    filename = sanitize_filename(url) + ".txt"
    output_path = output_dir / filename

    output_path.write_text(text, encoding="utf-8")

    print(f"  Saved -> {output_path}")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving files to:\n{output_dir}\n")

    for url in URLS:
        try:
            scrape_url(url, output_dir)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()