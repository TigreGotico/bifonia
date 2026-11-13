"""Tests for the learned statistical sense model and its integration."""
import sys

import pytest

from bifonia import tokenize, guess_sense
from bifonia.features import extract_features, load_structural_vocs
from bifonia.model import SenseModel, PERCEPTRON_PATH, NB_PATH

_HAS_MODEL = PERCEPTRON_PATH.exists()


def _idx(sentence, word):
    toks = tokenize(sentence.lower())
    # tokens may carry trailing punctuation — locate by the stripped form.
    i = next(j for j, t in enumerate(toks) if t.strip(".,;:!?") == word)
    return toks, i


# ── feature extractor ─────────────────────────────────────────────────────────

def test_features_are_deterministic():
    vocs = load_structural_vocs("pt-pt")
    toks, idx = _idx("A sede da empresa fica em Lisboa.", "sede")
    assert extract_features(toks, idx, vocs) == extract_features(toks, idx, vocs)


def test_features_capture_context_and_structure():
    vocs = load_structural_vocs("pt-pt")
    toks, idx = _idx("A sede da empresa fica em Lisboa.", "sede")
    feats = extract_features(toks, idx, vocs)
    assert feats["R1=da"] == 1.0            # positional skipgram
    assert "W=empresa" in feats             # bag-of-window
    assert feats.get("L1∈determiners") == 1.0  # structural membership ("a")


# ── model loading + prediction ───────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_MODEL, reason="trained model not present")
def test_model_loads_and_predicts_valid_sense():
    m = SenseModel.load(str(PERCEPTRON_PATH))
    toks, idx = _idx("A sede da empresa fica em Lisboa.", "sede")
    pred = m.predict("sede", toks, idx)
    assert pred in m.words["sede"]["senses"]


@pytest.mark.skipif(not _HAS_MODEL, reason="trained model not present")
def test_both_artifacts_share_schema():
    for path in (PERCEPTRON_PATH, NB_PATH):
        m = SenseModel.load(str(path))
        for entry in m.words.values():
            assert entry["route"] in ("model", "rules")
            assert set(entry["bias"]) == set(entry["senses"])


# ── integration: guess_sense uses the model but never regresses to a bad sense ─

def test_guess_sense_thirst_vs_seat():
    # sede routes to rules (overfit-guarded); both readings must resolve correctly
    for sentence, want in [("A sede da empresa fica em Lisboa.", "seat"),
                           ("Tinha tanta sede que bebi a garrafa toda.", "thirst")]:
        toks, idx = _idx(sentence, "sede")
        assert guess_sense(toks, idx) == want


def test_explicit_pos_override_uses_rules():
    toks, idx = _idx("Eu jogo às cartas todas as noites.", "jogo")
    assert guess_sense(toks, idx, pos="VERB") == "play"


# ── zero heavy dependencies on the inference path ─────────────────────────────

def test_no_heavy_deps_imported():
    """Disambiguation must not pull numpy / scipy / scikit-learn / torch (zero-dep
    install). Checked in a clean subprocess so pytest plugins can't pollute it."""
    import subprocess
    code = (
        "import sys; from bifonia import tokenize, guess_sense; "
        "w = tokenize('a sede da empresa fica em lisboa'); "
        "guess_sense(w, w.index('sede')); "
        "heavy = [m for m in ('numpy','scipy','sklearn','torch','onnxruntime') "
        "if m in sys.modules]; "
        "sys.exit('heavy deps pulled: ' + ','.join(heavy) if heavy else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
