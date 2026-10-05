"""Purpose: Build provider-specific chat requests and manage plain conversation turns.
Dependencies: built-in: none; installed: none.
Custom: none.
"""

CHAT_COMMANDS = {"/clear", "/bye", "/exit"}


def parse_chat_command(text):
    command = text.strip().casefold()
    return command if command in CHAT_COMMANDS else None


def build_request_messages(system_prompt, current_prompt, provider, history=None):
    messages = [{"role": "system", "content": system_prompt}]

    if provider.casefold() == "ollama" and history:
        messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in history
        )

    messages.append({"role": "user", "content": current_prompt})
    return messages


def record_completed_turn(history, provider, question, response):
    if provider.casefold() != "ollama" or not response or not response.strip():
        return

    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.strip()},
        ]
    )