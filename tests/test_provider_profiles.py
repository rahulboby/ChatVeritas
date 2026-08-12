import unittest
from unittest.mock import Mock, patch

from core.exceptions import ConfigurationError
from core.llm import DeployLLM, resolve_llm_provider
from core.utils import QuestionGenerator


def provider_config() -> dict:
    return {
        "generation": {"temperature": 0.3, "max_new_tokens": 64},
        "llm": {
            "provider": "groq",
            "temperature": 0.7,
            "max_retries": 3,
            "providers": {
                "groq": {
                    "model": "groq-model",
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key_env": "GROQ_API_KEY",
                    "requires_api_key": True,
                    "response_format": {"type": "json_object"},
                },
                "lmstudio": {
                    "model": "local-model",
                    "base_url": "http://localhost:1234/v1",
                    "api_key_env": "LMSTUDIO_API_KEY",
                    "default_api_key": "lm-studio",
                    "requires_api_key": False,
                    "response_format": None,
                },
            },
        },
    }


class ProviderProfileTests(unittest.TestCase):
    def test_lmstudio_alias_uses_local_profile_and_default_key(self):
        config = provider_config()
        config["llm"]["provider"] = "lm-studio"

        settings = resolve_llm_provider(config)

        self.assertEqual(settings.name, "lmstudio")
        self.assertEqual(settings.model, "local-model")
        self.assertEqual(settings.base_url, "http://localhost:1234/v1")
        self.assertEqual(settings.api_key, "lm-studio")
        self.assertIsNone(settings.response_format)

    def test_placeholder_local_model_is_rejected_before_client_creation(self):
        config = provider_config()
        config["llm"]["provider"] = "lmstudio"
        config["llm"]["providers"]["lmstudio"]["model"] = (
            "CHANGE_ME_TO_YOUR_LM_STUDIO_MODEL_ID"
        )

        with self.assertRaises(ConfigurationError):
            resolve_llm_provider(config)

    @patch("llm.deploy.OpenAI")
    def test_deploy_backend_uses_active_profile(self, openai_class):
        config = provider_config()
        config["llm"]["provider"] = "lmstudio"

        backend = DeployLLM(config)

        openai_class.assert_called_once_with(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1",
        )
        self.assertEqual(backend.provider.name, "lmstudio")

    @patch("utils.question_generator.OpenAI")
    def test_question_generator_omits_response_format_when_profile_disables_it(
        self, openai_class
    ):
        client = openai_class.return_value
        completion = Mock()
        completion.choices = [Mock(message=Mock(content='{"topic":"T","questions":["Q?"]}'))]
        client.chat.completions.create.return_value = completion

        generator = QuestionGenerator(
            api_key="lm-studio",
            model="local-model",
            base_url="http://localhost:1234/v1",
            response_format=None,
            max_retries=1,
        )

        result = generator.generate("A source paragraph.")

        self.assertEqual(result, {"topic": "T", "questions": ["Q?"]})
        self.assertNotIn(
            "response_format", client.chat.completions.create.call_args.kwargs
        )


if __name__ == "__main__":
    unittest.main()
