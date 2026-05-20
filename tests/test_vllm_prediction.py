from types import SimpleNamespace

import pytest
import torch

from src.vllm_prediction import VLLMTokenPredictor
from src.prediction import get_token_predictor


class FakeFlatLogprobs:
    def __init__(self, token_ids, logprobs):
        self.start_indices = [0]
        self.end_indices = [len(logprobs)]
        self.token_ids = token_ids
        self.logprobs = logprobs


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeCompletion:
    def __init__(self, logprobs):
        self.logprobs = logprobs


class FakeRequestOutput:
    def __init__(self, logprobs):
        self.outputs = [FakeCompletion(logprobs)]


class FakeSamplingParams:
    def __init__(self):
        self.extra_args = None

    def clone(self):
        return FakeSamplingParams()


def make_predictor(encoding="AC", reduce_tokens=False, tokens_list=None):
    predictor = VLLMTokenPredictor.__new__(VLLMTokenPredictor)
    predictor.vocab_size = 5
    predictor.tokens_list = tokens_list or list(range(predictor.vocab_size))
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = reduce_tokens
    predictor.device = torch.device("cpu")
    predictor.args = SimpleNamespace(encoding=encoding)
    predictor.max_batch_size = 32
    predictor.sampling_params = FakeSamplingParams()
    return predictor


def test_read_captured_intervals_returns_step_major_tensors(monkeypatch):
    import struct
    from multiprocessing import shared_memory

    predictor = make_predictor(encoding="AC_TARGET_INTERVAL")
    predictor.capture_vocab_size = 5
    predictor.max_vocab_cols = 8
    predictor._capture_record_stride = 32
    rows = 2
    steps = 2
    header_bytes = 24
    flags_offset = header_bytes
    data_offset = flags_offset + rows * steps * 4
    shm = shared_memory.SharedMemory(create=True, size=data_offset + rows * steps * 32)
    predictor._shm = shm
    try:
        struct.pack_into("IIIIII", shm.buf, 0, rows, steps, 8, 4, rows * steps, 1)
        records = {
            (0, 0): (1, 3, 10, 0.2),
            (0, 1): (2, 5, 10, 0.3),
            (1, 0): (3, 7, 10, 0.4),
            (1, 1): (4, 9, 10, 0.5),
        }
        for row in range(rows):
            for step in range(steps):
                flat = row * steps + step
                struct.pack_into("I", shm.buf, flags_offset + flat * 4, 1)
                struct.pack_into(
                    "qqqd",
                    shm.buf,
                    data_offset + flat * 32,
                    *records[(row, step)],
                )

        intervals = predictor._read_captured_intervals(rows, steps, squeeze=False)

        assert intervals["lows"].tolist() == [[1, 3], [2, 4]]
        assert intervals["highs"].tolist() == [[3, 7], [5, 9]]
        assert intervals["totals"].tolist() == [[10, 10], [10, 10]]
        assert torch.allclose(
            intervals["target_probs"],
            torch.tensor([[0.2, 0.4], [0.3, 0.5]], dtype=torch.float64),
        )
    finally:
        shm.close()
        shm.unlink()


def test_read_captured_logits_uses_strided_shared_memory():
    import struct
    from multiprocessing import shared_memory

    predictor = make_predictor(encoding="AC_MULTISTREAM")
    predictor.capture_vocab_size = 3
    predictor.max_vocab_cols = 4
    predictor._capture_record_stride = 32
    predictor.device = torch.device("cpu")
    rows = 2
    steps = 2
    cols = 4
    header_bytes = 24
    flags_offset = header_bytes
    data_offset = flags_offset + rows * steps * 4
    shm = shared_memory.SharedMemory(create=True, size=data_offset + rows * steps * 32)
    predictor._shm = shm
    try:
        struct.pack_into("IIIIII", shm.buf, 0, rows, steps, 4, cols, rows * steps, 0)
        records = {
            (0, 0): [0.0, 1.0, 2.0, 3.0],
            (0, 1): [4.0, 5.0, 6.0, 7.0],
            (1, 0): [8.0, 9.0, 10.0, 11.0],
            (1, 1): [12.0, 13.0, 14.0, 15.0],
        }
        for row in range(rows):
            for step in range(steps):
                flat = row * steps + step
                struct.pack_into("I", shm.buf, flags_offset + flat * 4, 1)
                row_array = torch.tensor(records[(row, step)], dtype=torch.float32)
                shm.buf[data_offset + flat * 32: data_offset + flat * 32 + cols * 4] = (
                    row_array.numpy().tobytes()
                )

        logits = predictor._read_captured_logits(rows, steps, squeeze=False)

        assert logits.shape == (steps, rows, 3)
        assert logits.tolist() == [
            [[0.0, 1.0, 2.0], [8.0, 9.0, 10.0]],
            [[4.0, 5.0, 6.0], [12.0, 13.0, 14.0]],
        ]
    finally:
        shm.close()
        shm.unlink()


def test_flat_logprobs_with_token_ids_to_dense_scores():
    predictor = make_predictor()
    logprobs = FakeFlatLogprobs(
        token_ids=[3, 1, 4],
        logprobs=[0.3, 0.1, 0.4],
    )

    row = predictor._position_logprobs_to_dense(logprobs, position=0)

    assert torch.isneginf(row[0])
    assert row.tolist()[1] == pytest.approx(0.1)
    assert row.tolist()[3] == pytest.approx(0.3)
    assert row.tolist()[4] == pytest.approx(0.4)


def test_flat_logprobs_values_only_to_dense_scores():
    predictor = make_predictor()
    logprobs = FakeFlatLogprobs(
        token_ids=[],
        logprobs=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    row = predictor._position_logprobs_to_dense(logprobs, position=0)

    assert row.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_dict_logprobs_to_dense_scores():
    predictor = make_predictor()
    logprobs = [{2: FakeLogprob(-2.0), 4: FakeLogprob(-4.0)}]

    row = predictor._position_logprobs_to_dense(logprobs, position=0)

    assert torch.isneginf(row[0])
    assert row.tolist()[2] == pytest.approx(-2.0)
    assert row.tolist()[4] == pytest.approx(-4.0)


def test_ac_reduced_scores_are_softmaxed_over_reduced_vocab(monkeypatch):
    predictor = make_predictor(
        encoding="AC",
        reduce_tokens=True,
        tokens_list=[1, 3],
    )
    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            return []

    predictor.llm = FakeLLM()
    predictor._reset_capture_buffer = lambda expected_rows, expected_steps: None
    predictor._read_captured_logits = lambda expected_rows, expected_steps, squeeze: torch.tensor(
        [
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )

    tokens_list, probs, _, _ = predictor.run_batched_inference([[7], [8]])

    assert tokens_list == [1, 3]
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    assert probs[0].tolist() == pytest.approx(
        torch.softmax(torch.tensor([1.0, 3.0]), dim=-1).tolist()
    )


def test_rank_encodings_return_raw_reduced_scores():
    predictor = make_predictor(
        encoding="bitpacked",
        reduce_tokens=True,
        tokens_list=[0, 4],
    )
    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            return []

    predictor.llm = FakeLLM()
    predictor._reset_capture_buffer = lambda expected_rows, expected_steps: None
    predictor._read_captured_logits = lambda expected_rows, expected_steps, squeeze: torch.tensor(
        [[0.0, 4.0]],
        dtype=torch.float32,
    )

    tokens_list, scores, _, softmax_time = predictor.run_batched_inference([[7]])

    assert tokens_list == [0, 4]
    assert scores.shape == (1, 2)
    assert scores[0].tolist() == pytest.approx([0.0, 4.0])
    assert softmax_time == 0.0


def test_get_token_predictor_dispatches_to_production_vllm(monkeypatch):
    created = []

    class FakeVLLMTokenPredictor:
        def __init__(self, args, bitmap_data):
            created.append((args, bitmap_data))

    monkeypatch.setattr(
        "src.vllm_prediction.VLLMTokenPredictor",
        FakeVLLMTokenPredictor,
    )

    args = SimpleNamespace(engine="vllm")
    predictor = get_token_predictor(args, bitmap_data=b"bitmap")

    assert isinstance(predictor, FakeVLLMTokenPredictor)
    assert created == [(args, b"bitmap")]
