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
    r"\bfrases (proibidas|excluídas)\b", r"perdoe, esta frase", r"\bRepito:",
    # tool-call and tokenizer artefacts
    r"</?arg_value>", r"<\|[^|]*\|>", r"`", r"/home/",
]))


# Removed rows, one per kind of meta-text, so the guard keeps catching each kind.
_KNOWN_META = [
    "Aqui tem 40 frases com «choco» adjetivo, em Português Europeu natural, "
    "abrangendo os vários sentidos e registos:",
    "Here are 40 sentences in European Portuguese using \"destroço\" as a verb:",
    "O esmero do我们的... — perdoe, esta frase não está em português. Repito: "
    "O esmero do sapateiro português era incomparável.",
    "Domo o leão que domina com força e coragem.</arg_value>",
    "Eu sossego o agoni<|reserved_token_16361|><|tool_call_begin|>ar com yoga.",
    "A sopa está muito quente para comer com esta coleres. (Wait, \"colher\" singular) "
    "-> A sopa está muito quente para comer com esta colher.",
]


def test_meta_pattern_matches_known_meta_rows():
    assert [s for s in _KNOWN_META if not _META.search(s)] == []


def test_corpus_has_no_generator_meta_text():
    hits = []
    for number, line in enumerate(_CORPUS.read_text(encoding="utf-8").splitlines(), 1):
        sentence = json.loads(line)["sentence"]
        if _META.search(sentence):
            hits.append((number, sentence[:80]))
    assert hits == []
