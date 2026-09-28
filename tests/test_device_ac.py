"""Device-side arithmetic state and inverse CDF must match the host decoder."""

import numpy as np
import pytest
import torch

from src.coding.device_ac import DeviceMultistreamACDecoder
from src.coding.multistream_ac import MultistreamACDecoder, MultistreamACEncoder
from src.coding.paired_ac import pack_standard_archive
from src.coding.target_interval import target_intervals_from_probs_tensor


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("paired", [False, True])
def test_device_decoder_matches_host_with_inactive_rows(device, paired):
    rng = np.random.default_rng(77)
    rows = rng.dirichlet(np.ones(13), size=(17, 3)).astype(np.float32)
    targets = rng.integers(0, 13, size=(17, 3))
    lengths = [17, 15, 9]
    encoder = MultistreamACEncoder(3, target_interval=True)
    for step in range(17):
        probabilities = torch.tensor(rows[step], device=device)
        target = torch.tensor(targets[step], device=device)
        low, high, total, _ = target_intervals_from_probs_tensor(probabilities, target)
        for stream, length in enumerate(lengths):
            if step < length:
                encoder.encode_interval(stream, int(low[stream]), int(high[stream]), int(total[stream]))
    standard = encoder.finish()
    archive = pack_standard_archive(standard) if paired else standard
    host = MultistreamACDecoder(standard)
    coder = DeviceMultistreamACDecoder(archive, device, paired=paired)
    for step in range(17):
        probabilities = torch.tensor(rows[step], device=device)
        active = [step < length for length in lengths]
        actual = coder.decode(probabilities, active).cpu().tolist()
        for stream in range(3):
            if active[stream]:
                expected = host.decode(stream, rows[step, stream])
                assert actual[stream] == expected == targets[step, stream]
    host.assert_complete()
    coder.assert_complete()
    with pytest.raises(ValueError, match="no more symbols"):
        coder.decode(torch.tensor(rows[-1], device=device), [True, False, False])
