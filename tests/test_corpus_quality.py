"""The bundled corpus holds sentences only, never generator meta-text."""
import json
import re
from pathlib import Path

_CORPUS = Path(__file__).parent.parent / "bifonia" / "data" / "corpus.jsonl"

_META = re.compile("|".join([
    # English generator chatter
    r"\bHere are\b", r"\bI'll\b", r"\bI've\b", r"\bLet me\b", r"\bI see\b",
    r"\bThe sentences\b", r"\bThese all\b", r"\bAll 40\b", r"\bWait\b",
    r"\bFile saved\b", r"\bsentences\b", r"\bphrases\b",
    # Portuguese generator chatter
    r"\bAqui (tem|estão) \d+", r"\bGerad[oa]s? \d+", r"\bJá (foram geradas|gerei)\b",
    r"\bVou gerar\b", r"\bGerando \d+", r"Verificação concluída", r"\bProntas —",
    r"\bEscrevi as \d+", r"\b\d+ frases (geradas|novas)\b", r"Nenhuma das frases",
    r"O diretório já", r"O usuário pede", r"Peço desculpa — o output",
    r"Preciso de corrigir", r"Deixa-me corrigir", r"Tarefa concluída", r"⚠",
    r"\bfrases (proibidas|excluídas)\b",
    # tool-call and tokenizer artefacts
    r"</?arg_value>", r"<\|[^|]*\|>", r"`", r"/home/",
]))


def test_corpus_has_no_generator_meta_text():
    hits = []
    for number, line in enumerate(_CORPUS.read_text(encoding="utf-8").splitlines(), 1):
        sentence = json.loads(line)["sentence"]
        if _META.search(sentence):
            hits.append((number, sentence[:80]))
    assert hits == []
