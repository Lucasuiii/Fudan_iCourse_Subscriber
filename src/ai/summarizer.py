"""LLM-based course lecture summarization with a versioned prompt file."""

import time
from pathlib import Path

from openai import OpenAI

from src.runtime import config
from src.ai.tavily_enrichment import enrich_summary

_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "lecture_summary.md"
)


def load_system_prompt(path: str | Path | None = None) -> str:
    """Load the DeepSeek-compatible system prompt from Markdown.

    Keeping the prompt outside Python makes writing changes reviewable without
    touching provider or pipeline logic.  Missing or empty prompts fail closed
    instead of silently sending the model an ungoverned request.
    """
    prompt_path = Path(path) if path is not None else _DEFAULT_PROMPT_PATH
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Summary prompt is empty: {prompt_path}")
    return prompt


class Summarizer:
    """Course lecture summarizer with multi-provider fallback.

    Iterates config.MODEL_PROVIDERS in declared order. Within each provider,
    tries each model in declared order. Returns the first successful result.
    Setting only DASHSCOPE_API_KEY still works because the default
    MODEL_PROVIDERS list ships a modelscope entry that reads it.
    """

    def __init__(self):
        self.system_prompt = load_system_prompt()
        self.providers = config.resolve_model_providers()
        if not self.providers:
            raise ValueError(
                "No model provider available. "
                "Set at least one provider's API key (e.g. DASHSCOPE_API_KEY)."
            )
        self._clients = {
            p["name"]: OpenAI(api_key=p["api_key"], base_url=p["base_url"])
            for p in self.providers
        }

    def _call_llm(self, client: OpenAI, model: str,
                  title: str, content: str) -> str:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"以下是课程《{title}》的原始材料。请沿能够确认"
                        "的授课脉络，将其整理为详细、连贯、适合复习的"
                        "课程笔记。\n\n"
                        f"<course_material>\n{content}\n</course_material>"
                    ),
                },
            ],
            temperature=0.2,
            timeout=180,
        )
        if not response.choices:
            raise ValueError("API returned empty choices — likely content filter or quota exceeded")
        result = response.choices[0].message.content
        elapsed = time.time() - t0
        # Token usage helps explain run cost — every provider's billing is
        # token-based, and rate-limit decisions key off prompt size much
        # more than character count.  Some providers (OpenAI-compatible)
        # leave usage None on streaming or error paths, so fall back to a
        # plain "no usage" line so the summary still prints.
        usage = getattr(response, "usage", None)
        if usage is not None:
            print(
                f"[Summarizer] Done ({model}): "
                f"{len(content)} chars input → {len(result)} chars output"
                f" in {elapsed:.0f}s "
                f"(tokens: prompt={getattr(usage,'prompt_tokens','?')}, "
                f"completion={getattr(usage,'completion_tokens','?')})"
            )
        else:
            print(
                f"[Summarizer] Done ({model}): {len(content)} chars input"
                f" → {len(result)} chars output in {elapsed:.0f}s"
            )
        return result

    def summarize(self, title: str, content: str) -> tuple[str, str]:
        """Summarize lecture, trying providers in MODEL_PROVIDERS order.

        Returns (summary, model_used) where model_used is "{provider}/{model}".

        Raises:
            RuntimeError: if all providers/models fail.
        """
        if not content or not content.strip():
            return ("（内容为空）", "")

        errors = []
        for provider in self.providers:
            client = self._clients[provider["name"]]
            for model in provider["models"]:
                model_id = f"{provider['name']}/{model}"
                try:
                    result = self._call_llm(client, model, title, content)
                    result = enrich_summary(
                        result,
                        api_key=config.TAVILY_API_KEY,
                        client=client,
                        model=model,
                    )
                    return (result, model_id)
                except Exception as e:
                    print(f"[Summarizer] {model_id} failed: "
                          f"{type(e).__name__}")
                    errors.append(f"{model_id}: {e}")

        raise RuntimeError(
            "All LLM models failed:\n" + "\n".join(errors)
        )
