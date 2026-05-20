# test for prediction.py

from types import SimpleNamespace
import pytest
import torch
from tqdm import tqdm

from src.prediction import TokenPredictor

@pytest.fixture
def test_setup():
    args = SimpleNamespace(
        model_name="gpt2",
        engine="transformer",
        reduce_tokens=True,
        encoding="AC",
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        spec_k=3,
    )
    predictor = TokenPredictor(args=args, bitmap_data=None)
    prompts = [[1, 2, 3], [4, 5, 6]]
    return predictor, prompts


def test_run_batched_inferences(test_setup, print_on = False):
    """Test the standard batched inference method separately."""
    dummy_predictor, dummy_prompts = test_setup

    tokens_list, scores, data_copy_time, softmax_time = dummy_predictor.run_batched_inference(dummy_prompts)

    if print_on:
        print("Tokens list shape:", len(tokens_list))
        print("Most likely next token ids:", [tokens_list[torch.argmax(scores[i]).item()] for i in range(scores.shape[0])])
        print("Scores shape:", scores.shape)
        print("\n")

    assert isinstance(tokens_list, list), "Output tokens should be a list"
    assert all(isinstance(tok, int) for tok in tokens_list), "Each item in tokens list should be an integer token ID"
    assert isinstance(scores, torch.Tensor), "Scores should be a torch.Tensor"
    assert scores.dim() == 2, f"Scores should be a 2D tensor with shape (batch_size, vocab_size or reduced_vocab_size). Is {scores.shape} but expected (len(dummy_prompts), len(self.tokens_list))"


def test_run_batched_inference_cachefree(test_setup, print_on = True):
    """Test the cachefree inference method separately."""
    dummy_predictor, dummy_prompts = test_setup

    # run cachefree inference for testing
    tokens_list_cf, scores_cf, data_copy_time_cf, softmax_time_cf = dummy_predictor.run_batched_inference_cachefree(dummy_prompts)

    if print_on:
        print("Cachefree - Tokens list shape:", len(tokens_list_cf))
        print("Cachefree - Most likely next token ids:", [tokens_list_cf[torch.argmax(scores_cf[i]).item()] for i in range(scores_cf.shape[0])])
        print("Cachefree - Scores shape:", scores_cf.shape)

    assert isinstance(tokens_list_cf, list), "Output tokens should be a list"
    assert all(isinstance(tok, int) for tok in tokens_list_cf), "Each item in tokens list should be an integer token ID"
    assert isinstance(scores_cf, torch.Tensor), "Scores should be a torch.Tensor"
    assert scores_cf.dim() == 2, f"Scores should be a 2D tensor with shape (batch_size, vocab_size or reduced_vocab_size). Is {scores_cf.shape} but expected (len(dummy_prompts), len(self.tokens_list))"


def test_run_batched_inference_cachefree_uneven_input(test_setup, print_on = True):
    """Test the cachefree inference method with uneven input lengths --> needed for draft generation."""
    dummy_predictor, _ = test_setup

    dummy_prompts = [[1, 2], [3, 4, 5, 6]]  # uneven length prompts

    # run cachefree inference for testing
    tokens_list_cf, scores_cf, data_copy_time_cf, softmax_time_cf = dummy_predictor.run_batched_inference_cachefree(dummy_prompts)

    if print_on:
        print("Cachefree - Most likely next token ids:", [tokens_list_cf[torch.argmax(scores_cf[i]).item()] for i in range(scores_cf.shape[0])])
        print("Cachefree - Scores shape:", scores_cf.shape)
        print("Cachefree - Data copy time:", data_copy_time_cf)
        print("Cachefree - Softmax time:", softmax_time_cf)
        print("\n")

    assert all(isinstance(tok, int) for tok in tokens_list_cf), "Each item in tokens list should be an integer token ID"
    assert isinstance(scores_cf, torch.Tensor), "Scores should be a torch.Tensor"
    assert scores_cf.dim() == 2, "Scores should be a 2D tensor with shape (batch_size, vocab_size or reduced_vocab_size)"


def test_run_batched_inferences_alignment(test_setup):
    """Test that the standard and cachefree inference methods produce the same outputs when kv_cache is enabled."""
    dummy_predictor, dummy_prompts = test_setup

    tokens_list, scores, _, _ = dummy_predictor.run_batched_inference(dummy_prompts, enable_kv_cache=True)
    tokens_list_cf, scores_cf, _, _ = dummy_predictor.run_batched_inference_cachefree(dummy_prompts)

    print("Standard vs Cachefree - Tokens list shape:", len(tokens_list), "vs", len(tokens_list_cf))
    print("Standard vs Cachefree - Scores shape:", scores.shape, "vs", scores_cf.shape)

    assert len(tokens_list) == len(tokens_list_cf), "Tokens list length mismatch between standard and cachefree inference"
    assert scores.shape == scores_cf.shape, "Scores shape mismatch between standard and cachefree inference"


@pytest.mark.parametrize("k", [1, 3])
def test_generate_draft(test_setup, k, print_on = False):
    dummy_predictor, dummy_prompts = test_setup

    draft_tokens, draft_scores, total_data_copy_time, total_softmax_time = dummy_predictor.generate_draft(dummy_prompts, k=k)

    if print_on:
        print("Draft tokens:", draft_tokens)
        print("Draft scores shape:", draft_scores.shape)  # should be (k, batch_size, vocab_size_or_reduced_vocab_size)
        print("Total data copy time for draft generation:", total_data_copy_time)
        print("Total softmax time for draft generation:", total_softmax_time)

    assert isinstance(draft_tokens, list)
    assert len(draft_tokens) == len(dummy_prompts)
    assert all(len(row) == k for row in draft_tokens), f"Expected {k} draft tokens per batch item, got {draft_tokens}"
    assert isinstance(draft_scores, torch.Tensor)
    assert draft_scores.shape[0] == k
    assert draft_scores.shape[1] == len(dummy_prompts)

    print("test_generate_draft passed.\n")


@pytest.mark.parametrize("k", [2])
def test_generate_draft_correctness(test_setup, k, print_on = True):
    """Do two steps of draft generation manually and verify that the output matches the generate_draft method."""
    dummy_predictor, dummy_prompts = test_setup

    draft_tokens, draft_scores, total_data_copy_time, total_softmax_time = dummy_predictor.generate_draft(dummy_prompts, k=k)
    B = len(dummy_prompts)
    if print_on:
        print('Test generate_draft correctness with k=2')
        print("Draft tokens:", draft_tokens)
        print("Draft scores shape:", draft_scores.shape)  # should be (k, batch_size, vocab_size_or_reduced_vocab_size)
        print("Total data copy time for draft generation:", total_data_copy_time)
        print("Total softmax time for draft generation:", total_softmax_time)

    # do steps manually to verfiy correctness of draft generation
    manual_draft_tokens = []

    for i in tqdm(range(k)):
        tokens_list, scores, _, _ = dummy_predictor.run_batched_inference_cachefree(dummy_prompts)
        most_likely_tokens = [tokens_list[torch.argmax(scores[j]).item()] for j in range(scores.shape[0])]
        manual_draft_tokens.append(most_likely_tokens) # store tokens for each step separately for easier debugging
        print(f"Manual draft tokens for step {i}:", most_likely_tokens)

        # Update prompts for next step using the tokens produced in this step
        dummy_prompts = [dummy_prompts[j] + [most_likely_tokens[j]] for j in range(B)]
        print(f"Updated prompts after step {i}:", dummy_prompts)

    # Transpose manual_draft_tokens to match the shape of draft_tokens
    manual_draft_tokens = list(map(list, zip(*manual_draft_tokens)))

    assert draft_tokens == manual_draft_tokens, f"Draft tokens mismatch. Expected {manual_draft_tokens}, got {draft_tokens}"

    print("test_generate_draft passed.\n")


@pytest.mark.parametrize("k", [1, 3])
def test_generate_draft_uneven_input(test_setup, k, print_on = True):
    dummy_predictor, dummy_prompts = test_setup
    dummy_prompts = [[1, 2], [3, 4, 5, 6]] # uneven length prompts

    draft_tokens, draft_scores, total_data_copy_time, total_softmax_time = dummy_predictor.generate_draft(dummy_prompts, k=k)

    print("Draft tokens:", draft_tokens)
    print("Draft scores shape:", draft_scores.shape)  # should be (k, batch_size, vocab_size_or_reduced_vocab_size)
    print("Total data copy time for draft generation:", total_data_copy_time)
    print("Total softmax time for draft generation:", total_softmax_time)

    assert isinstance(draft_tokens, list)
    assert len(draft_tokens) == len(dummy_prompts)
    assert all(len(row) == k for row in draft_tokens), f"Expected {k} draft tokens per batch item, got {draft_tokens}"
    assert isinstance(draft_scores, torch.Tensor)
    assert draft_scores.shape[0] == k
    assert draft_scores.shape[1] == len(dummy_prompts)

    print("test_generate_draft_uneven_input passed.\n")


def test_reset_kv_cache(test_setup):
    dummy_predictor, dummy_prompts = test_setup

    # populate cache first
    _ = dummy_predictor.run_batched_inference(dummy_prompts, enable_kv_cache=True)

    assert hasattr(dummy_predictor, "_past_kv")
    assert hasattr(dummy_predictor, "_cached_context_len")

    print("Before reset:")
    print("Has _past_kv:", dummy_predictor._past_kv is not None)
    print("_cached_context_len:", dummy_predictor._cached_context_len)

    dummy_predictor.reset_kv_cache()

    print("After reset:")
    print("Has _past_kv:", dummy_predictor._past_kv is not None)
    print("_cached_context_len:", dummy_predictor._cached_context_len)

    assert dummy_predictor._past_kv is None
    assert dummy_predictor._cached_context_len == 0

    print("test_reset_kv_cache passed.\n")


def test_greedy_next_token(test_setup):
    dummy_predictor, dummy_prompts = test_setup

    prompt = dummy_prompts[0]

    next_token, score_vec, data_copy_time, softmax_time = dummy_predictor.greedy_next_token(
        prompt,
        enable_kv_cache=False,
    )

    print("Prompt:", prompt)
    print("Greedy next token:", next_token)
    print("Score vector shape:", score_vec.shape)
    print("Data copy time:", data_copy_time)
    print("Softmax time:", softmax_time)

    assert isinstance(next_token, int)
    assert isinstance(score_vec, torch.Tensor)
    assert score_vec.dim() == 1

    print("test_greedy_next_token passed.\n")

@pytest.mark.parametrize("k", [1, 3])
def test_accept_reject_loop(test_setup, k):
    dummy_predictor, dummy_prompts = test_setup

    # First generate a draft from the same model.
    # Since draft + verifier are currently the same model,
    # this will often accept all drafted tokens.
    draft_tokens, draft_scores, draft_data_copy_time, draft_softmax_time = dummy_predictor.generate_draft(
        dummy_prompts,
        k=k,
    )

    print("Input prompts:", dummy_prompts)
    print("Draft tokens:", draft_tokens)

    (   accepted_tokens,
        accepted_lengths,
        updated_prompts,
        rejected,
        total_data_copy_time,
        total_softmax_time,
    ) = dummy_predictor.accept_reject_loop(dummy_prompts, draft_tokens)

    print("Accepted tokens:", accepted_tokens)
    print("Accepted lengths:", accepted_lengths)
    print("Updated prompts:", updated_prompts)
    print("Rejected flags:", rejected)
    print("Verifier data copy time:", total_data_copy_time)
    print("Verifier softmax time:", total_softmax_time)

    assert isinstance(accepted_tokens, list)
    assert isinstance(accepted_lengths, list)
    assert isinstance(updated_prompts, list)
    assert isinstance(rejected, list)

    assert len(accepted_tokens) == len(dummy_prompts)
    assert len(accepted_lengths) == len(dummy_prompts)
    assert len(updated_prompts) == len(dummy_prompts)
    assert len(rejected) == len(dummy_prompts)

    for i in range(len(dummy_prompts)):
        assert accepted_lengths[i] == len(accepted_tokens[i])
        assert len(updated_prompts[i]) >= len(dummy_prompts[i])

    print("test_accept_reject_loop passed.\n")

def test_accept_reject_loop_forced_rejection(test_setup):
    """
    Test the accept/reject loop with a forced mismatch to verify that rejections are handled correctly.
    Mismatch is forced between first token in first batch element. Rest of batches align with the draft to isolate the effect of a single mismatch.
    """
    dummy_predictor, dummy_prompts = test_setup
    true_tokens, _, _, _ = dummy_predictor.generate_draft(dummy_prompts, k=3)

    # Force a mismatch in the first batch element by corrupting the first draft token
    draft_tokens = [row[:] for row in true_tokens] # deep copy to avoid modifying original true_tokens
    
    # assuming token IDs are integers, this will create a mismatch with the verifier's prediction for the second token in the first batch element
    draft_tokens[0][1] = draft_tokens[0][1] + 1 

    print("Original / True draft tokens:", true_tokens)
    print("Draft tokens:", draft_tokens)

    (   accepted_tokens,
        accepted_lengths,
        updated_prompts,
        rejected,
        total_data_copy_time,
        total_softmax_time,
    ) = dummy_predictor.accept_reject_loop(dummy_prompts, draft_tokens)

    print("Accepted tokens:", accepted_tokens)
    print("Accepted lengths:", accepted_lengths)
    print("Updated prompts:", updated_prompts)
    print("Rejected flags:", rejected)

    assert rejected[0] is True, "Expected forced rejection in batch element 0"
    assert accepted_lengths[0] == 1, f"Expected one accepted token before first forced mismatch, got {accepted_lengths[0]}"

    print("test_accept_reject_loop_forced_rejection passed.\n")


@pytest.mark.integration
@pytest.mark.parametrize("k", [1, 3])
def test_speculative_decode(test_setup, k):
    dummy_predictor, dummy_prompts = test_setup

    max_new_tokens = 5

    final_prompts, generated_tokens = dummy_predictor.speculative_decode(
        dummy_prompts,
        max_new_tokens=max_new_tokens,
        k=k,)

    print("Original prompts:", dummy_prompts)
    print("Generated tokens:", generated_tokens)
    print("Final prompts:", final_prompts)

    assert isinstance(final_prompts, list)
    assert isinstance(generated_tokens, list)
    assert len(final_prompts) == len(dummy_prompts)
    assert len(generated_tokens) == len(dummy_prompts)

    for i in range(len(dummy_prompts)):
        assert len(generated_tokens[i]) == max_new_tokens, (
            f"Expected {max_new_tokens} generated tokens, got {len(generated_tokens[i])}"
        )
        assert final_prompts[i] == dummy_prompts[i] + generated_tokens[i], (
            f"Final prompt mismatch for batch element {i}"
        )

    print("test_speculative_decode passed.\n")
