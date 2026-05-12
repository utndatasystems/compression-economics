"""Helpers for deciding whether a cached prompt state can advance incrementally."""


def freeze_prompt(prompt_tokens):
    return tuple(int(token_id) for token_id in prompt_tokens)


def freeze_prompts(prompts):
    return tuple(freeze_prompt(prompt_tokens) for prompt_tokens in prompts)


def prompt_extends_one_token(previous_prompt, current_prompt):
    previous = freeze_prompt(previous_prompt)
    current = freeze_prompt(current_prompt)
    return len(current) == len(previous) + 1 and current[:-1] == previous


def prompts_extend_one_token(previous_prompts, current_prompts):
    if previous_prompts is None:
        return False

    current = freeze_prompts(current_prompts)
    if len(previous_prompts) != len(current):
        return False

    return all(
        len(current_prompt) == len(previous_prompt) + 1 and current_prompt[:-1] == previous_prompt
        for previous_prompt, current_prompt in zip(previous_prompts, current)
    )