from research.papers.coding_survey.experiments.benchmark_speculative_full import (
    OnlineNgram,
    replay,
    split_streams,
)


def test_split_streams_preserves_order_and_balances_lengths():
    streams = split_streams(list(range(11)), 3)
    assert streams == [list(range(4)), list(range(4, 8)), list(range(8, 11))]


def test_online_ngram_does_not_learn_unconfirmed_tokens():
    model = OnlineNgram(max_order=3)
    history = [1]
    model.unigram.update(history)
    assert model.propose(history, 2) == [1, 1]
    model.observe(history, [2, 3])
    assert history == [1, 2, 3]
    assert model.tables[1][(1,)][2] == 1
    assert model.tables[2][(1, 2)][3] == 1


def test_replay_recovers_source_and_reduces_calls_on_repetition():
    streams = [[7, 8, 7, 8, 7, 8, 7, 8], [7, 8, 7, 8, 7, 8]]
    result = replay(streams, draft_length=2, max_order=2, scope="shared")
    assert result["exact_source_recovery"] is True
    assert result["target_call_reduction"] > 1
    assert result["accepted_draft_tokens"] > 0


def test_per_stream_replay_handles_uneven_rows():
    result = replay([[1, 2, 3], [4, 5]], 4, 2, "per_stream")
    assert result["exact_source_recovery"] is True
    assert result["baseline_synchronous_target_calls"] == 2
