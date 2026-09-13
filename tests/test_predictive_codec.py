from src.relational_compression_benchmark.predictive_codec import compress_and_verify
from src.models import NGramPredictor
from src.predictors import train_ngram_predictor


def test_predictive_arithmetic_codec_round_trips_with_reconstructed_context():
    predictor = NGramPredictor(3, order=2)
    train_ngram_predictor(predictor, [0, 1, 2, 0, 1, 2])
    result = compress_and_verify(predictor, [0, 1, 2, 0, 1, 2])

    assert result.roundtrip_valid
    assert result.payload_bits > 0
    assert result.encode_seconds >= 0
    assert result.decode_seconds >= 0
