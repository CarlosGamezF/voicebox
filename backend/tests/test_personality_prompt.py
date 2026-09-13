"""The personality rewrite must not change the language of the input.

A small Qwen3 model given an English character description happily restates
Spanish input in English, and the rewritten text is then synthesised with
``language="es"``: the clone reads English with Spanish conditioning, which
surfaces as a heavy accent. The rewrite task now pins the output language to
the input's language.
"""

from backend.services.personality import _REWRITE_TASK, _build_system_prompt


def test_rewrite_task_keeps_the_input_language():
    assert "same language as the user's text" in _REWRITE_TASK
    assert "never translate" in _REWRITE_TASK


def test_rewrite_system_prompt_carries_the_language_rule():
    prompt = _build_system_prompt("A grumpy old sailor from Cádiz.", _REWRITE_TASK)

    assert "same language as the user's text" in prompt
    assert "A grumpy old sailor from Cádiz." in prompt
