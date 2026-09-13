import math

import pytest
import torch

from src.models import NGramPredictor, TinyRecurrentLM
from src.predictors import (
    ModelSpec,
    bits_per_symbol,
    build_predictor,
    survey_model_specs,
    train_ngram_predictor,
    train_neural_predictor,
)


def test_survey_has_requested_byte_and_token_ngrams_and_windows():
    specs = {spec.name: spec for spec in survey_model_specs()}

    assert {"byte-bigram", "byte-trigram", "token-bigram", "token-trigram"} <= specs.keys()
    assert {specs[f"token-nplm-w{window}"].context_length for window in (8, 32, 128)} == {8, 32, 128}
    assert specs["token-tiny-transformer-w128"].family == "transformer"
    assert specs["token-tiny-gru-w128"].family == "recurrent"


def test_trigram_backs_off_and_prefers_seen_continuation():
    predictor = NGramPredictor(4, order=3)
    train_ngram_predictor(predictor, [0, 1, 2, 0, 1, 2])

    seen = predictor.logits([[0, 1]])
    unseen = predictor.logits([[3, 3]])

    assert seen.shape == (1, 4)
    assert seen.argmax(dim=-1).item() == 2
    assert torch.isfinite(unseen).all()
    assert math.isfinite(bits_per_symbol(predictor, [0, 1, 2]))


@pytest.mark.parametrize("family", ["nplm", "transformer", "recurrent"])
def test_neural_predictors_accept_short_contexts(family):
    spec = ModelSpec("test", family, "token", 8, embedding_dim=8, hidden_dim=16, layers=1, heads=2)
    predictor = build_predictor(spec, vocabulary_size=7)

    logits = predictor.logits([[], [1], [1, 2, 3]])

    assert logits.shape == (3, 7)
    assert torch.isfinite(logits).all()


def test_neural_training_is_teacher_forced_and_returns_epoch_losses():
    model = TinyRecurrentLM(3, context_length=4, embedding_dim=4, hidden_dim=8, layers=1)

    losses = train_neural_predictor(
        model, [0, 1, 2] * 8, epochs=2, batch_size=6, learning_rate=0.01, seed=4
    )

    assert len(losses) == 2
    assert all(math.isfinite(loss) for loss in losses)
    assert bits_per_symbol(model, [0, 1, 2, 0]) > 0


def test_byte_models_are_pinned_to_the_byte_vocabulary():
    with pytest.raises(ValueError, match="vocabulary_size=256"):
        build_predictor(survey_model_specs()[0], vocabulary_size=7)
