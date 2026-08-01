"""
scripts/chat.py

Terminal interface for ChatVeritas (offline mode).

Instantiates the ChatVeritas application interface and runs an interactive REPL.
All retrieval, reranking, prompt assembly, and model inference are
handled transparently by the application modules — this script
contains no pipeline logic of its own.
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so application modules are
# importable regardless of the working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Thread limits before any heavy library imports.
# ---------------------------------------------------------------------------
from core.constants import apply_thread_limits
apply_thread_limits()

from core.logger import get_logger
from interfaces.chatveritas import ChatVeritas

logger = get_logger(__name__)


def main() -> None:
    overall_start = time.perf_counter()

    print("=" * 80)

    choice = input("Use LoRA? (y/n): ").lower().strip()
    use_lora = choice in ["y", "yes"]

    logger.info("Starting ChatVeritas terminal (offline | use_lora=%s).", use_lora)

    try:
        chatbot = ChatVeritas(mode="offline", use_lora=use_lora)
    except Exception as e:
        logger.error("Failed to initialise ChatVeritas: %s", e, exc_info=True)
        sys.exit(1)

    startup_time = time.perf_counter() - overall_start
    logger.info("Startup complete in %.2f s.", startup_time)

    print(f"\nStartup time: {startup_time:.2f} s")
    print("\nRAG Chat Ready")
    print("Type 'exit' to quit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            result = chatbot.ask(question)
            response = result["response"]
        except Exception as e:
            logger.error("Error during generation: %s", e, exc_info=True)
            print(f"\n[Error] {e}\n")
            continue

        print("\nAssistant:\n")
        print(response)
        print()


if __name__ == "__main__":
    main()
