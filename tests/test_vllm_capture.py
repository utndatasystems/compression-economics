"""Exercise the real vLLM capture processor with a local scheduler substitute."""

import struct
from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch

from src.vllm_prediction import BatchedLogitsCaptureProcessor, VLLMTokenPredictor
from src.coding.multistream_ac import MultistreamACEncoder, MultistreamACDecoder


class SamplingParams:
    def __init__(self):
        self.extra_args = {}

    def clone(self):
        return SamplingParams()


@pytest.mark.parametrize("masked", [False, True])
def test_reordered_worker_capture_is_compact_and_decodable(masked):
    predictor = VLLMTokenPredictor.__new__(VLLMTokenPredictor)
    predictor.args = SimpleNamespace(encoding="AC_TARGET_INTERVAL")
    predictor.tokens_list = [1, 3] if masked else list(range(4))
    predictor.capture_vocab_size = len(predictor.tokens_list)
    predictor.max_vocab_cols = 4
    predictor.max_batch_size = 2
    predictor.max_window_size = 1
    predictor.sampling_params = SamplingParams()
    predictor._interval_only = True
    predictor._shm = shared_memory.SharedMemory(create=True, size=24 + 2 * 4 + 2 * 32)
    worker = BatchedLogitsCaptureProcessor.__new__(BatchedLogitsCaptureProcessor)
    worker._shm = predictor._shm
    worker._select_capture_logits = lambda logits: logits[:, predictor.tokens_list]
    worker._target_column_index = lambda token: predictor.tokens_list.index(token)
    logits = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])

    class Engine:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            assert [prompt["prompt_token_ids"] for prompt in prompts] == [[1], [3]]
            # Internal batch ordering differs from caller row ordering.
            worker.req_info = {
                0: {"row_id": 1, "output_token_ids": [],
                    "target_token_ids": sampling_params[1].extra_args["ce_target_token_ids"]},
                1: {"row_id": 0, "output_token_ids": [],
                    "target_token_ids": sampling_params[0].extra_args["ce_target_token_ids"]},
            }
            forced_logits = worker.apply(logits.clone())
            assert forced_logits.argmax(dim=1).tolist() == [3, 1]

    predictor.llm = Engine()
    try:
        _, intervals, _, _ = predictor.run_batched_interval_inference([[1], [3]], [1, 3])
        assert predictor.last_interval_transfer_bytes == 64
        encoder = MultistreamACEncoder(2, target_interval=True)
        for row in range(2):
            encoder.encode_interval(row, int(intervals["lows"][row]),
                                    int(intervals["highs"][row]), int(intervals["totals"][row]))
        decoder = MultistreamACDecoder(encoder.finish())
        probabilities = torch.softmax(logits.flip(0)[:, predictor.tokens_list], dim=-1)
        assert [predictor.tokens_list[decoder.decode(row, probabilities[row].numpy())]
                for row in range(2)] == [1, 3]
        decoder.assert_complete()

        # An unready slot must never be read as a valid interval.
        struct.pack_into("I", predictor._shm.buf, 24, 0)
        with pytest.raises(RuntimeError, match="Missing captured"):
            predictor._read_captured_intervals(2, 1, squeeze=True)
    finally:
        predictor.cleanup()


def test_predictor_dispatch_is_lazy(monkeypatch):
    from src.global_mask_compressor import _make_token_predictor
    args = SimpleNamespace(engine="vllm")
    monkeypatch.setattr("src.vllm_prediction.VLLMTokenPredictor",
                        lambda args, bitmap: (args.engine, bitmap))
    assert _make_token_predictor(args, b"bitmap") == ("vllm", b"bitmap")
