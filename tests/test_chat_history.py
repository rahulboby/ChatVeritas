"""Purpose: Verify provider-specific conversation requests, history, and commands.
Dependencies: built-in: contextlib, io, types, unittest; installed: none.
Custom: scripts.chat, utils.chat_history.
"""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.chat import generate_response, init_chatveritas, main
from utils.chat_history import (
    build_request_messages,
    parse_chat_command,
    record_completed_turn,
)


class ChatHistoryTests(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.retriever = Mock()
        self.retriever.retrieve.return_value = {
            "results": [
                {
                    "chunk": "retrieved private context",
                    "source": "guide.txt",
                    "chunk_id": 1,
                    "distance": 0.1,
                }
            ],
            "metrics": {},
        }
        self.config = {
            "active_llm": {"model": "test-model", "provider": "ollama"},
            "generation": {"temperature": 0.2, "max_new_tokens": 20},
        }

    @staticmethod
    def stream_chunk(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
        )

    def test_cli_stores_plain_question_after_complete_stream(self):
        self.client.chat.completions.create.return_value = [
            self.stream_chunk("A response")
        ]
        history = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]

        with redirect_stdout(StringIO()):
            response = generate_response(
                "Current plain question",
                self.client,
                self.retriever,
                self.config,
                history,
            )

        request = self.client.chat.completions.create.call_args.kwargs
        self.assertIn("retrieved private context", request["messages"][-1]["content"])
        self.assertEqual(response, "A response")
        self.assertEqual(
            history[-2:],
            [
                {"role": "user", "content": "Current plain question"},
                {"role": "assistant", "content": "A response"},
            ],
        )
        self.assertNotIn("retrieved private context", str(history))

    def test_general_chat_sends_direct_api_request_without_retrieval(self):
        self.client.chat.completions.create.return_value = [
            self.stream_chunk("A general answer")
        ]
        history = []

        with redirect_stdout(StringIO()):
            response = generate_response(
                "Explain the idea simply",
                self.client,
                None,
                self.config,
                history,
            )

        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("general-purpose assistant", messages[0]["content"])
        self.assertIn("Explain the idea simply", messages[-1]["content"])
        self.assertNotIn("retrieved private context", str(messages))
        self.assertEqual(response, "A general answer")
        self.assertEqual(history[-2]["content"], "Explain the idea simply")

    def test_general_chat_is_available_without_saved_vectorstores(self):
        config = {
            "model_ollama": {
                "url": "http://localhost:11434/v1",
                "provider": "ollama",
                "model": "test-model",
            },
            "model_openai": {
                "url": "https://api.openai.com/v1",
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
        }
        with (
            unittest.mock.patch("scripts.chat.load_dotenv"),
            unittest.mock.patch("scripts.chat.load_config", return_value=config),
            unittest.mock.patch("scripts.chat.list_vectorstores", return_value=[]),
            unittest.mock.patch("scripts.chat.OpenAI") as openai_client,
            unittest.mock.patch("builtins.input", side_effect=["0", ""]),
            redirect_stdout(StringIO()) as output,
        ):
            client, retriever, loaded_config = init_chatveritas()

        self.assertIsNone(retriever)
        self.assertEqual(client, openai_client.return_value)
        self.assertEqual(loaded_config["active_llm"], config["model_ollama"])
        self.assertIn("0. General chatting", output.getvalue())

    def test_cli_openai_endpoint_uses_configured_url_and_env_key(self):
        config = {
            "model_openai": {
                "url": "https://api.openai.com/v1",
                "provider": "openai",
                "model": "gpt-4o-mini",
            }
        }
        with (
            unittest.mock.patch("scripts.chat.load_dotenv"),
            unittest.mock.patch("scripts.chat.load_config", return_value=config),
            unittest.mock.patch("scripts.chat.list_vectorstores", return_value=[]),
            unittest.mock.patch("scripts.chat.os.getenv", return_value="test-openai-key"),
            unittest.mock.patch("scripts.chat.OpenAI") as openai_client,
            unittest.mock.patch(
                "builtins.input",
                side_effect=["0", "oa"],
            ) as input_mock,
            redirect_stdout(StringIO()),
        ):
            client, retriever, loaded_config = init_chatveritas()

        openai_client.assert_called_once_with(
            api_key="test-openai-key",
            base_url="https://api.openai.com/v1",
        )
        self.assertEqual(client, openai_client.return_value)
        self.assertIsNone(retriever)
        self.assertEqual(loaded_config["active_llm"], config["model_openai"])
        self.assertIn("OpenAI (oa)", input_mock.call_args_list[1].args[0])

    def test_cli_openai_endpoint_requires_openai_api_key(self):
        config = {
            "model_openai": {
                "url": "https://api.openai.com/v1",
                "provider": "openai",
                "model": "gpt-4o-mini",
            }
        }
        with (
            unittest.mock.patch("scripts.chat.load_dotenv"),
            unittest.mock.patch("scripts.chat.load_config", return_value=config),
            unittest.mock.patch("scripts.chat.list_vectorstores", return_value=[]),
            unittest.mock.patch("scripts.chat.os.getenv", return_value=None),
            unittest.mock.patch("builtins.input", side_effect=["0", "openai"]),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                init_chatveritas()

    def test_cli_does_not_store_partial_response_on_stream_failure(self):
        def broken_stream():
            yield self.stream_chunk("partial")
            raise RuntimeError("stream interrupted")

        self.client.chat.completions.create.return_value = broken_stream()
        history = []

        with redirect_stdout(StringIO()):
            response = generate_response(
                "Current question",
                self.client,
                self.retriever,
                self.config,
                history,
            )

        self.assertIsNone(response)
        self.assertEqual(history, [])

    def test_cli_clear_resets_history_before_next_question(self):
        history_snapshots = []

        def fake_generate_response(question, client, retriever, config, history):
            history_snapshots.append(list(history))
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "answer"},
                ]
            )

        with (
            unittest.mock.patch(
                "scripts.chat.init_chatveritas",
                return_value=(
                    self.client,
                    self.retriever,
                    self.config,
                ),
            ),
            unittest.mock.patch(
                "scripts.chat.generate_response",
                side_effect=fake_generate_response,
            ),
            unittest.mock.patch(
                "builtins.input",
                side_effect=["first question", "/clear", "second question", "/bye"],
            ),
            redirect_stdout(StringIO()) as output,
        ):
            main()

        self.assertEqual(history_snapshots, [[], []])
        self.assertIn("Conversation history cleared.", output.getvalue())

    def test_cli_reports_openai_history_as_stateless(self):
        config = {
            "active_llm": {"model": "test-model", "provider": "openai"},
            "generation": {"temperature": 0.2, "max_new_tokens": 20},
        }
        with (
            unittest.mock.patch(
                "scripts.chat.init_chatveritas",
                return_value=(self.client, None, config),
            ),
            unittest.mock.patch("builtins.input", side_effect=["/clear", "/bye"]),
            redirect_stdout(StringIO()) as output,
        ):
            main()

        self.assertIn("OpenAI conversation history is disabled.", output.getvalue())
        self.assertIn("OpenAI does not retain conversation history.", output.getvalue())

    def test_ollama_request_contains_plain_prior_turns_and_current_prompt(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        messages = build_request_messages(
            "System prompt",
            "Current question with retrieved context",
            "ollama",
            history,
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "System prompt"},
                *history,
                {"role": "user", "content": "Current question with retrieved context"},
            ],
        )

    def test_groq_request_omits_prior_turns(self):
        messages = build_request_messages(
            "System prompt",
            "Current question with retrieved context",
            "groq",
            [{"role": "user", "content": "Old question"}],
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Current question with retrieved context"},
            ],
        )

    def test_record_completed_turn_stores_only_plain_ollama_exchange(self):
        history = []
        record_completed_turn(history, "ollama", "Plain question", "Answer")
        record_completed_turn(history, "groq", "Groq question", "Groq answer")
        record_completed_turn(history, "ollama", "Failed question", "")

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "Plain question"},
                {"role": "assistant", "content": "Answer"},
            ],
        )

    def test_commands_are_case_insensitive_and_exact(self):
        self.assertEqual(parse_chat_command(" /CLEAR "), "/clear")
        self.assertEqual(parse_chat_command("/Bye"), "/bye")
        self.assertEqual(parse_chat_command("/exit now"), None)
        self.assertEqual(parse_chat_command("normal question"), None)


if __name__ == "__main__":
    unittest.main()