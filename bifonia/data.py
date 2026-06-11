"""
IPA mappings and context wordlists for Portuguese heterophonic bifoniaaph disambiguation.

IPA data is loaded from tugalex's heterophonic_homographs.csv (sibling package).
"""

import csv
import pathlib

_CSV = pathlib.Path(__file__).parent / "data" / "heterophonic_homographs.csv"

# word → {POS: IPA}  e.g. {"para": {"ADP": "ˈpɐɾɐ", "VERB": "ˈpaɾɐ"}}
HOMOGRAPHS: dict = {}
with _CSV.open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        HOMOGRAPHS.setdefault(row["word"], {})[row["pos"]] = row["ipa"]

# "sobre" ADP (about/concerning/over) shares IPA with NOUN (envelope) — the CSV
# only carries NOUN and VERB entries, so inject the ADP reading explicitly.
HOMOGRAPHS["sobre"]["ADP"] = HOMOGRAPHS["sobre"]["NOUN"]

# derived per-POS dicts (backward-compatible)
VERBS_IPA: dict = {w: d["VERB"] for w, d in HOMOGRAPHS.items() if "VERB" in d}
NOUNS_IPA: dict = {w: d["NOUN"] for w, d in HOMOGRAPHS.items() if "NOUN" in d}
ADP_IPA: dict = {w: d["ADP"] for w, d in HOMOGRAPHS.items() if "ADP" in d}
ADJ_IPA: dict = {w: d["ADJ"] for w, d in HOMOGRAPHS.items() if "ADJ" in d}

AMBIGUOUS_WORDS: set = set(HOMOGRAPHS)

# sensible linguistic defaults when context scoring yields no winner
DEFAULT_POS: dict = {
    "para": "ADP",
    "pelo": "ADP",
    "sobre": "ADP",
    "seco": "ADJ",
    "tola": "ADJ",
    "porto": "NOUN",
    "acordo": "NOUN",
    "acerto": "NOUN",
    "cerro": "NOUN",
    "choro": "NOUN",
    "colher": "NOUN",
    "começo": "NOUN",
    "conserto": "NOUN",
    "coro": "NOUN",
    "corte": "NOUN",
    # "forma" NOUN IPA (closed-o) = baking mold only (rare).  All other NOUN
    # senses (manner, shape, geometric form) share the open-o VERB IPA.
    # Default to VERB so "uma forma geométrica" / "desta forma" get open-o.
    "forma": "VERB",
    "gozo": "NOUN",
    "gosto": "NOUN",
    "jogo": "NOUN",
    "molho": "NOUN",
    "olho": "NOUN",
    "rego": "NOUN",
    "sede": "NOUN",
    "torre": "NOUN",
    "transtorno": "NOUN",
    "peso": "NOUN",
    "posto": "NOUN",
}
assert set(DEFAULT_POS) == AMBIGUOUS_WORDS, (
    f"DEFAULT_POS missing: {AMBIGUOUS_WORDS - set(DEFAULT_POS)}"
)

# ── per-word POS bias (European Portuguese frequency priors) ──────────────────
# Integer bonus added to each POS score before signal scoring begins.
# Use to encode corpus-level frequency priors when the default tiebreaker
# (DEFAULT_POS) is not granular enough.  Tune by running benchmark_tagger.py
# and inspecting per-word accuracy.  Values are intentionally small (1–3) so
# they lose to any explicit context signal.
BASE_SCORE: dict = {
    # Small frequency priors to break near-ties.
    # Keep values ≤ 1 to avoid overriding any explicit context signal.
    # "sobre" ADP prior is intentionally 0: the DET-before+7 NOUN boost already
    # gives score_noun=12 for unambiguous envelope sentences, and a non-zero ADP
    # prior would create ties that degrade NOUN accuracy.
    # "pelo" most commonly ADP (por+o) in EP prose — +1 to break bare-context ties.
    "pelo": {"ADP": 1, "NOUN": 0, "VERB": 0},
    # three-way and two-way words: no safe non-zero prior without corpus tuning.
}

# ── semantic wordlists for specific words ─────────────────────────────────────

# "para" VERB ("parar" = to stop) — things that stop in European Portuguese.
# If any of these appears as the SUBJECT (nearby prev nouns) or OBJECT after
# "para", the VERB reading "pára/para" (stops) is strongly supported.
# European Portuguese usage: vehicles, machines, bodily functions, processes.
# Easy to extend: add the canonical EP noun for any stoppable entity.
STOPPABLE_THINGS = {
    # vehicles & transport (EP names)
    "autocarro", "comboio", "metro", "eléctrico", "elétrico",
    "carro", "automóvel", "viatura", "caminhão", "camião", "veículo",
    "barco", "navio", "avião", "helicóptero", "mota", "bicicleta",
    "escada", "rolante", "tapete",  # escalators/conveyor belts
    # machines & equipment
    "máquina", "motor", "bomba", "gerador", "ventilador", "turbina",
    "impressora", "computador", "servidor", "relógio",
    "correia", "transportadora", "elevador", "grua",
    # bodily / biological processes
    "coração", "hemorragia", "sangramento", "sangue", "pulsação",
    "respiração", "choro", "tosse", "vómito",
    # flows & processes
    "música", "música", "som", "sinal", "transmissão", "emissão",
    "obra", "obras", "construção", "produção", "actividade", "atividade",
    "narração", "gravação", "reprodução", "música", "som",
    "chuva", "neve", "granizo", "vento",
    # abstract stops
    "guerra", "conflito", "violência", "disputa", "greve",
    "funcionamento", "serviço", "atendimento",
}

# ── context wordlists ─────────────────────────────────────────────────────────

DET = {
    "o", "a", "os", "as",
    "um", "uma", "uns", "umas",
    "este", "esta", "estes", "estas",
    "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas",
    "meu", "minha", "meus", "minhas",
    "teu", "tua", "teus", "tuas",
    "seu", "sua", "seus", "suas",
    "do", "da", "dos", "das",
    # contracted forms: "a+o/a/os/as" — treated as DET so NOUN scoring fires
    "ao", "à", "aos", "às",
    # contracted "em+article": "no/na/nos/nas" — appear before nouns in
    # locative phrases ("no sobre", "na forma", "no porto"), strong NOUN signal
    "no", "na", "nos", "nas",
    # contracted "em+indefinite": "num/numa/nuns/numas" — same locative pattern
    # ("num sobre", "numa forma", "num porto"), also NOUN signal
    "num", "numa", "nuns", "numas",
}

# Quantifier determiners: precede noun phrases like DET but are NOT included in
# AFTER_PREP (to avoid giving ADP an inflated +5 bonus for "sobre algum X").
# Used in score_noun and score_adj attributive checks.
QUANT = {
    "nenhum", "nenhuma", "nenhuns", "nenhumas",
    "todo", "toda", "todos", "todas",
    "cada",
    "algum", "alguma", "alguns", "algumas",
    "outro", "outra", "outros", "outras",
    "certo", "certa", "certos", "certas",
    "qualquer", "quaisquer",
    "pouco", "pouca", "poucos", "poucas",
    "muito", "muita", "muitos", "muitas",
    "tanto", "tanta", "tantos", "tantas",
}

PRON = {
    "eu", "tu", "ele", "ela",
    "nós", "vós", "eles", "elas",
    "me", "te", "se", "nos", "vos",
    "quem", "que", "qual",
}

PREP = {"para", "pelo", "sobre"}

AUX_VERBS = {
    "vou", "vais", "vai", "vamos", "vão",
    "fui", "foste", "foi", "fomos", "fostes", "foram",
    "ficar", "ficou", "ficam", "fica",
}

NUMERIC = {
    "um", "uma", "uns", "umas", "dois", "duas",
    "três", "quatro", "cinco", "seis", "sete", "oito", "nove",
    "dez", "onze", "doze", "treze", "catorze", "quinze", "dezasseis",
    "dezassete", "dezoito", "dezanove", "vinte", "trinta", "quarenta",
    "cinquenta", "sessenta", "setenta", "oitenta", "noventa", "cem",
    "duzentos", "duzentas", "trezentos", "trezentas", "quatrocentos",
    "mil", "milhões", "triliões",
}

BEFORE_PREP = {"parte"} | AUX_VERBS

# Nouns and verbs that strongly govern "sobre" as a preposition meaning
# "about/concerning".  Presence of any of these immediately before "sobre"
# is a reliable ADP signal (e.g. "caso sobre X", "falou sobre X").
SOBRE_GOV = {
    # governing nouns ("caso" excluded: doubles as conditional conjunction + subjunctive VERB,
    # e.g. "caso sóbre comida" = "if food is left over"; plural "casos" is unambiguous)
    "casos", "artigo", "artigos", "livro", "livros", "livrete",
    "debate", "debates", "discussão", "discussões", "opinião", "opiniões",
    "informação", "informações", "notícia", "notícias", "relatório", "relatórios",
    "dados", "evidência", "evidências", "estudo", "estudos", "análise", "análises",
    "investigação", "investigações", "pesquisa", "pesquisas", "texto", "textos",
    "documento", "documentos", "declaração", "declarações", "comentário", "comentários",
    "reflexão", "reflexões", "pergunta", "perguntas", "dúvida", "dúvidas",
    "conferência", "conferências", "palestra", "palestras", "seminário", "seminários",
    "apresentação", "apresentações", "consenso", "unanimidade", "acordo",
    "posição", "posições", "ponto", "pontos", "questão", "questões",
    "problema", "problemas", "tema", "temas", "assunto", "assuntos",
    "situação", "situações", "aspeto", "aspetos", "aspecto", "aspectos",
    "direito", "direitos", "lei", "leis", "regra", "regras",
    # publications, media, documents, talks that introduce "sobre" as ADP
    "manual", "manuais", "monografia", "monografias", "capítulo", "capítulos",
    "biografia", "biografias", "blog", "blogs", "podcast", "podcasts",
    "simpósio", "simpósios", "colóquio", "colóquios", "congresso", "congressos",
    "vídeo", "vídeos", "filme", "filmes", "série", "séries", "documentário", "documentários",
    "reportagem", "reportagens", "entrevista", "entrevistas", "crónica", "crónicas",
    "poema", "poemas", "conto", "contos", "romance", "romances", "ensaio", "ensaios",
    "teoria", "teorias", "hipótese", "hipóteses", "tese", "teses", "dissertação", "dissertações",
    # event/course/workshop framing nouns that introduce "sobre" as ADP
    "workshop", "workshops", "webinar", "webinars", "curso", "cursos",
    "módulo", "módulos", "unidade", "unidades", "formação", "formações",
    "programa", "programas", "projeto", "projetos", "iniciativa", "iniciativas",
    "tratado", "tratados", "acordo", "acordos", "protocolo", "protocolos",
    "convênio", "convênios", "convenção", "convenções", "resolução", "resoluções",
    "questionário", "questionários", "inquérito", "inquéritos", "sondagem", "sondagens",
    "inquérito", "relatório", "exposição", "exposições", "mostra", "mostras",
    "campanha", "campanhas", "projeto", "projetos", "ação", "ações",
    "legislação", "regulamento", "regulamentos", "norma", "normas", "decreto", "decretos",
    "proposta", "propostas", "recomendação", "recomendações", "diretiva", "diretivas",
    # governing verbs (3rd-person or infinitive forms common in context)
    "falar", "fala", "falou", "falamos", "falam",
    "escrever", "escreve", "escreveu", "escrevemos", "escrevem",
    "pensar", "pensa", "pensou", "pensamos", "pensam",
    "saber", "sabe", "soube", "sabemos", "sabem",
    "aprender", "aprende", "aprendeu", "aprendemos", "aprendem",
    "ensinar", "ensina", "ensinou", "ensinamos", "ensinam",
    "discutir", "discute", "discutiu", "discutimos", "discutem",
    "concordar", "concorda", "concordou",
    "discordar", "discorda", "discordou",
    "decidir", "decide", "decidiu",
    "conversar", "conversa", "conversou",
    "refletir", "reflete", "refletiu",
    "unanimidade", "consenso",
}

AFTER_PREP = (
    {"amanhã", "ontem", "depois", "sempre", "logo", "já", "quando", "todos", "ti", "mim",
     # interrogative + relative pronouns following prepositions
     "quê", "quem", "qual", "quais",
     # possessive determiners (seu/teu/meu/nosso/vosso): "pelo seu bem", "pelo meu cálculo"
     "meu", "minha", "meus", "minhas",
     "teu", "tua", "teus", "tuas",
     "seu", "sua", "seus", "suas",
     "nosso", "nossa", "nossos", "nossas",
     "vosso", "vossa", "vossos", "vossas",
     # common Portuguese cities / destinations (lowercased; proper nouns lost after tokenize)
     "lisboa", "porto", "coimbra", "braga", "faro", "évora", "setúbal", "viseu", "aveiro",
     "sintra", "cascais", "almada", "funchal", "ponta", "angra", "horta",
     "madrid", "paris", "berlim", "roma", "londres", "amsterdam", "bruxelas",
     "brasil", "angola", "moçambique", "cabo", "guiné", "portugal", "espanha",
    }
    | DET | PRON
) - {"do", "da", "dos", "das", "no", "na", "nos", "nas",
     # contracted "em+article" forms signal NOUN, not valid after ADP "para/pelo/sobre"
     "num", "numa", "nuns", "numas"}
# contracted article forms are NOUN signals, not valid ADP complements

NEVER_AFTER_PREP = {"ao", "aos", "à", "às", "no", "na", "nos", "nas"}
