"""Purpose: Crawl pages from seed URLs and save extracted text as local files.
Dependencies: built-in: pathlib, re, urllib.parse, collections, sys; installed: requests, beautifulsoup4, trafilatura.
Custom: none.
"""

from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import trafilatura

# ==============================================================================
# EDIT THESE SETTINGS
# ==============================================================================

URLS = [
    "https://devdocs.io/openjdk~8/",
    "https://docs.oracle.com/javase/8/docs/"
]

MAX_DEPTH = 1

OUTPUT_DIR = "data/raw/"

TIMEOUT = 10

# ==============================================================================


def sanitize_filename(name: str) -> str:
    """Convert a URL into a safe filesystem name."""
    name = re.sub(r"https?://", "", name)
    name = re.sub(r"[^\w\-_.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:150]


def fetch_and_extract(url: str):
    """
    Download the page and extract its main content as Markdown.
    Returns (content_text, BeautifulSoup_soup) or (None, None) on failure.
    """
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"  ❌ Failed to download {url}: {e}")
        return None, None

    text = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )

    soup = BeautifulSoup(html, "html.parser")
    return text, soup


def extract_links(base_url: str, soup: BeautifulSoup):
    """Return a set of absolute HTTP/HTTPS URLs from all <a> tags on the page."""
    links = set()
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        if not href or not isinstance(href, str):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme in ("http", "https"):
            cleaned = parsed._replace(fragment="").geturl()
            links.add(cleaned)
    return links


def save_content(url: str, content: str, output_dir: Path):
    """Write extracted content to a .txt file named after the URL."""
    if not content:
        return
    filename = sanitize_filename(url) + ".txt"
    out_path = output_dir / filename
    out_path.write_text(content, encoding="utf-8")
    print(f"  💾 Saved -> {out_path}")


def bfs_crawl(seed_urls, max_depth, output_dir):
    """
    Breadth‑first crawl starting from seed_urls.
    For each visited page, fetch content and save it.
    """
    visited = set()
    queue = deque()
    for url in seed_urls:
        if url not in visited:
            queue.append((url, 0))
            visited.add(url)

    total_processed = 0

    while queue:
        current_url, depth = queue.popleft()
        print(f"\n🔍 [{depth}] {current_url}")

        content, soup = fetch_and_extract(current_url)
        if content:
            save_content(current_url, content, output_dir)
        else:
            print("  ⚠️  No content extracted.")

        total_processed += 1

        if depth < max_depth and soup is not None:
            new_links = extract_links(current_url, soup)
            for link in new_links:
                if link not in visited:
                    visited.add(link)
                    queue.append((link, depth + 1))

    print(f"\n✅ Done. Processed {total_processed} pages.")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving files to: {output_dir}\n")

    import sys

    # Parse command-line arguments
    seeds = []
    depth = MAX_DEPTH  # default

    if len(sys.argv) > 1:
        # Collect all arguments that are URLs or files
        for arg in sys.argv[1:]:
            if arg.startswith("http"):
                seeds.append(arg)
            elif Path(arg).is_file():
                with open(arg, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and line.startswith("http"):
                            seeds.append(line)
        # If the last argument is a number, treat it as depth override
        if len(sys.argv) > 2 and sys.argv[-1].isdigit():
            depth = int(sys.argv[-1])
    else:
        # No command-line args: use the global defaults
        seeds = URLS

    # If seeds is still empty, fallback to defaults (shouldn't happen, but safeguard)
    if not seeds:
        seeds = URLS

    bfs_crawl(seeds, depth, output_dir)


if __name__ == "__main__":
    main()