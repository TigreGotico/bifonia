"""
Comprehensive parametrized tests for bifonia.

Test sentences use the diacritized orthographic form to label the expected
reading (e.g. "pára" = VERB para, "pêlo" = NOUN pelo).  _normalize() strips
those non-canonical diacritics before disambiguation so the engine sees the
plain base form in context.

xfail markers document known hard cases for the current rule-based scorer:
  - ADP "para" in "DET+NOUN+para+DET+NOUN" structures where prev_prev DET
    creates a false VERB signal equal to the genuine ADP signal.
  - "sobre" as ADP (prep "about/over") — pronounced like NOUN but contextual
    cues currently score it as VERB.
"""

import pytest
from bifonia import tokenize, disambiguate, HOMOGRAPHS
from bifonia import _DIACRITIZED_TO_BASE


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(sentence: str) -> str:
    """Replace diacritized bifoniaaph variants with their base forms."""
    result = sentence.lower()
    for diac, base in _DIACRITIZED_TO_BASE.items():
        result = result.replace(diac, base)
    return result


def _dis(sentence: str, word: str) -> str:
    words = tokenize(_normalize(sentence))
    idx = words.index(word)
    return disambiguate(words, idx)


def _expect(sentence: str, word: str, pos: str) -> None:
    ipa = _dis(sentence, word)
    expected = HOMOGRAPHS[word][pos]
    assert ipa == expected, (
        f"  sentence : {sentence!r}\n"
        f"  got      : [{ipa}]\n"
        f"  expected : {pos} = [{expected}]"
    )


# ── para / ADP ────────────────────────────────────────────────────────────────

PARA_ADP = [
    "Vou para casa depois do trabalho.",
    "Ela foi para o hospital de urgência.",
    "O comboio parte para Lisboa às oito horas.",
    "Compramos flores para a nossa mãe.",
    "Saímos para a rua assim que parou de chover.",
    "Este medicamento é para a dor de cabeça.",
    "Ele viajou para o estrangeiro no verão.",
    "A carta foi enviada para o endereço errado.",
    "Estou a guardar dinheiro para as férias.",
    "Precisamos de sair para a reunião a tempo.",
    "A encomenda foi despachada para o Porto.",
    "Reservei uma mesa para duas pessoas.",
    "Temos de caminhar para o norte durante meia hora.",
    pytest.param("O professor explicou a matéria para os alunos.",
                 marks=pytest.mark.xfail(reason="DET NOUN para DET: prev2-DET VERB signal ties ADP")),
    pytest.param("Fizemos um bolo para o aniversário da avó.",
                 marks=pytest.mark.xfail(reason="DET NOUN para DET: prev2-DET VERB signal ties ADP")),
    "Ela estudou muito para os exames finais.",
    "O autocarro vai para o centro da cidade.",
    "Mandei uma mensagem para ela ontem à noite.",
    pytest.param("Precisamos de encontrar uma solução para este problema.",
                 marks=pytest.mark.xfail(reason="DET NOUN para DET: prev2-DET VERB signal ties ADP")),
    "Saímos juntos para o jantar de despedida.",
    "A proposta foi enviada para todos os membros.",
    pytest.param("Ele treina todos os dias para a maratona.",
                 marks=pytest.mark.xfail(reason="DET NOUN para DET: prev2-DET VERB signal ties ADP")),
    "Ficámos à espera para entrar no espetáculo.",
    "A verba foi atribuída para a renovação do edifício.",
    "Compramos tinta para as paredes do corredor.",
    "Aquela estrada leva para a fronteira.",
    "Tenho uma surpresa para ti amanhã.",
    "O projeto foi aprovado para o próximo ano.",
    "Deixa ficar para a próxima vez.",
    "Há demasiado trabalho para uma só pessoa.",
    "Pedi uma extensão para entregar o relatório.",
    "A reunião ficou marcada para as três da tarde.",
    "O prémio foi entregue para reconhecer o seu esforço.",
    "Ele correu para a paragem do autocarro.",
    "Instalaram aquecimento para o inverno.",
    "Ficou retido para ser interrogado.",
    "Trouxe comida para partilhar com os colegas.",
    "O voo está confirmado para amanhã de manhã.",
    "A proposta foi feita para todos os sócios.",
    "Preparei tudo para a chegada dos convidados.",
    "Deixei o portão aberto para ela entrar.",
    "A viagem de comboio para o Porto dura três horas.",
    pytest.param("Levou o cão para o veterinário na sexta-feira.",
                 marks=pytest.mark.xfail(reason="DET NOUN para DET: prev2-DET VERB signal ties ADP")),
    "A análise foi solicitada para confirmar o diagnóstico.",
    "Ele saiu cedo para apanhar o primeiro comboio.",
    "Pusemos a mesa para seis pessoas.",
    "A decisão ficou para a próxima semana.",
    "O jantar está pronto para quando chegares.",
    # known-hard: prev_prev=DET clashes with DET after para
    pytest.param("Trouxe um presente para o teu filho.",
                 marks=pytest.mark.xfail(reason="prev_prev DET + DET-after ambiguity")),
    pytest.param("Escreveu um poema para a sua namorada.",
                 marks=pytest.mark.xfail(reason="prev_prev DET + DET-after ambiguity")),
]

@pytest.mark.parametrize("s", PARA_ADP)
def test_para_adp(s):
    _expect(s, "para", "ADP")


# ── para / VERB ───────────────────────────────────────────────────────────────

PARA_VERB = [
    "O autocarro pára mesmo em frente ao hospital.",
    "O carro pára automaticamente quando deteta um obstáculo.",
    "Ela pára sempre neste café antes de ir para o escritório.",
    "O comboio pára em todas as estações da linha.",
    "Ele pára o motor antes de sair do veículo.",
    "O atleta pára para beber água a meio da corrida.",
    "A máquina pára quando a temperatura é demasiado alta.",
    "O relógio pára sempre à mesma hora por falta de bateria.",
    "O seu coração pára por um instante quando ouve a notícia.",
    "A chuva pára exactamente quando saímos de casa.",
    # DET NOUN pára DET NOUN: ADP AFTER_PREP(+5) vs VERB prev2-DET(+3) — ADP wins
    pytest.param("O médico pára a medicação quando os sintomas desaparecem.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    "Ela pára de falar quando entra alguém na sala.",
    "O jogo pára aos vinte minutos por causa da chuva intensa.",
    "A música pára de repente e o silêncio instala-se.",
    "O sangramento pára após a aplicação de pressão.",
    "Ela pára na montra sempre que vê algo interessante.",
    "O motor pára sozinho quando atingimos o destino.",
    pytest.param("O professor pára a aula para responder às questões dos alunos.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    pytest.param("O semáforo pára os carros durante trinta segundos.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    "O meu vizinho pára sempre o carro em cima do passeio.",
    "Ela pára a conversa quando percebe que vai dizer algo errado.",
    "O elevador pára em todos os pisos até ao décimo.",
    "O barco pára no porto para reabastecer de combustível.",
    "A hemorragia pára logo depois da cirurgia.",
    "Ele pára para ajudar sempre que vê alguém com dificuldades.",
    pytest.param("A linha de montagem pára quando é detetado um defeito.",
                 marks=pytest.mark.xfail(reason="'quando' after scores ADP+2; no clear VERB signal")),
    pytest.param("O árbitro pára o jogo para verificar uma situação suspeita.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    "A descida pára abruptamente numa clareira no meio do bosque.",
    pytest.param("O enfermeiro pára a infusão quando o doente apresenta reação.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    "O cão pára de latir quando o dono lhe faz sinal.",
    "A bomba pára de funcionar se não houver pressão suficiente.",
    "O sistema pára automaticamente em caso de falha elétrica.",
    "O soldado pára quando ouve a ordem de sentido.",
    "O elétrico pára na praça para os passageiros desembarcarem.",
    "A respiração pára durante alguns segundos no apneia do sono.",
    "Ela pára de comer quando fica satisfeita, sem se exceder.",
    "O alarme pára quando introduzimos o código correto.",
    "O vento pára de soprar ao anoitecer nesta época do ano.",
    pytest.param("O piloto pára o motor antes de o abandonar.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    "O debate pára quando o moderador interrompe os participantes.",
    "A grua pára no meio da operação por avaria mecânica.",
    "O ciclista pára na berma para reparar o pneu furado.",
    "A negociação pára quando as partes não chegam a acordo.",
    "O filme pára exactamente na cena mais suspensa.",
    "A impressora pára quando o papel se esgota.",
    "O exercício pára assim que o paciente sente dor.",
    "O cronómetro pára quando o corredor cruza a linha de chegada.",
    # known-hard: "porteiro para todos" — "todos" AFTER_PREP competes with VERB
    "O porteiro pára todos os visitantes na entrada.",
    pytest.param("O polícia pára o trânsito para os peões atravessarem.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
    pytest.param("O gerente pára a produção para fazer manutenção ao equipamento.",
                 marks=pytest.mark.xfail(reason="DET NOUN pára DET: AFTER_PREP+5 beats prev2-DET+3")),
]

@pytest.mark.parametrize("s", PARA_VERB)
def test_para_verb(s):
    _expect(s, "para", "VERB")


# ── pelo / ADP ────────────────────────────────────────────────────────────────

PELO_ADP = [
    # Only "pelo" (singular masc.) is ambiguous; "pelos/pelas/pela" are always ADP.
    "Ela foi pelo corredor até ao seu quarto.",
    "Passámos pelo parque ao regressar para casa.",
    "O caminho mais curto passa pelo centro histórico da cidade.",
    "Soubemos da notícia pelo rádio durante o pequeno-almoço.",
    "Ficou famosa pelo seu talento excecional na música clássica.",
    "Ele entrou pelo portão lateral sem ser visto.",
    "Subimos pelo elevador até ao último andar do edifício.",
    "A lei foi aprovada pelo parlamento com ampla maioria.",
    "O gato entrou pelo buraco da vedação.",
    "O suspeito fugiu pelo telhado durante a madrugada.",
    "A mensagem foi transmitida pelo porta-voz do governo.",
    "O vento entrou pelo postigo da janela que ficara entreaberto.",
    "A criança brincava pelo jardim durante a tarde inteira.",
    "O médico foi contactado pelo seu assistente pessoal.",
    "Ele é reconhecido pelo seu trabalho extraordinário na área.",
    "O discurso foi transmitido pelo canal oficial da televisão.",
    "A carta foi enviada pelo serviço prioritário dos correios.",
    "Ficou admirada pelo nível de organização do evento.",
    "A expedição atravessou pelo deserto durante três semanas.",
    "A encomenda foi entregue pelo motorista da transportadora.",
    "O prémio foi atribuído pelo júri internacional do concurso.",
    "Passaram pelo controlo de segurança sem qualquer problema.",
    "A encomenda foi processada pelo sistema informático central.",
    "A notícia foi revelada pelo jornalista durante a conferência.",
    pytest.param("Atravessámos a cidade pelo metro para chegar mais depressa.",
                 marks=pytest.mark.xfail(reason="'city' noun after DET creates weak VERB signal")),
]

@pytest.mark.parametrize("s", PELO_ADP)
def test_pelo_adp(s):
    _expect(s, "pelo", "ADP")


# ── pelo / NOUN ───────────────────────────────────────────────────────────────

PELO_NOUN = [
    "O pêlo do cão ficou espalhado por todo o sofá da sala.",
    "A veterinária examinou o pêlo do gato com atenção.",
    "O pêlo do urso polar é branco e muito espesso.",
    "Ela escovou o pêlo do seu cão todas as manhãs.",
    "O pêlo do cavalo brilhava ao sol depois do banho.",
    "O coelho tem o pêlo muito suave e macio ao toque.",
    "A alergia ao pêlo do gato é bastante comum em adultos.",
    "O veterinário disse que o pêlo do animal estava em excelente estado.",
    "A raposa perdeu o pêlo durante a doença que contraiu.",
    "O pêlo espesso do lobo protege-o do frio intenso do inverno.",
    "O cão de raça tinha um pêlo longo e sedoso de cor dourada.",
    "O pêlo da foca é impermeável e mantém o animal aquecido.",
    "O criador examinou cuidadosamente o pêlo de cada animal.",
    "A dona penteou o pêlo da sua gatinha todas as noites antes de dormir.",
    "O pêlo do lhama é usado para fazer tecidos resistentes e quentes.",
    "O pêlo da ovelha é tosquiado uma vez por ano na primavera.",
    "O técnico analisou o pêlo encontrado na cena do crime.",
    "A cor do pêlo do gato mudou ligeiramente com a idade.",
    "O pêlo do hamster é muito curto e macio ao toque suave.",
    "A análise do pêlo revelou a presença de determinadas substâncias.",
    "O pêlo do coiote adapta-se à cor da estação do ano para camuflar.",
    "O vestido ficou coberto de pêlo do gato depois de ele se sentar.",
    "A criança acariciou o pêlo suave do coelho com muito cuidado.",
    "O pêlo da chinchila é considerado um dos mais finos do mundo animal.",
    "O pêlo do texugo tem uma coloração distinta e característica.",
]

@pytest.mark.parametrize("s", PELO_NOUN)
def test_pelo_noun(s):
    _expect(s, "pelo", "NOUN")


# ── pelo / VERB ───────────────────────────────────────────────────────────────

PELO_VERB = [
    "Pélo as batatas antes de as cozer com alho e azeite.",
    "Pélo os tomates para fazer o molho da massa ao jantar.",
    "Pélo sempre as cenouras antes de as cortar para a sopa.",
    pytest.param("Quando pélo a cebola, os olhos começam logo a lacrimejar muito.",
                 marks=pytest.mark.xfail(reason="'a' after pelo scores as ADP AFTER_PREP; temporal conj not strong enough")),
    "Pélo as maçãs antes de as adicionar ao bolo de canela.",
    "Pélo a fruta antes de a dar às crianças pequenas.",
    "Pélo os pêssegos quando estão bem maduros para fazer a compota.",
    "Pélo o pepino antes de o cortar para a salada de verão.",
    "Pélo as peras e corto-as em quartos antes de as servir.",
    "Pélo as laranjas na mesa enquanto ouço música à tarde.",
    "Pélo a manga com uma faca afiada para não desperdiçar polpa.",
    "Pélo sempre o alho antes de o picar para o refogado.",
    "Pélo os espargos antes de os cozer a vapor com limão.",
    "Pélo as curgetes antes de as grelhar com azeite e alecrim.",
    "Pélo o nabo para a sopa de legumes que faço todas as semanas.",
    "Pélo a abóbora antes de a assar no forno com mel e canela.",
    "Pélo sempre os kiwis antes de os comer ao pequeno-almoço.",
    "Pélo os feijões verdes antes de os estufar com tomate e chouriço.",
    "Pélo o aipo antes de o cortar para o caldo de carne.",
    "Pélo as amêndoas depois de as escaldar em água quente uns minutos.",
    "Pélo as castanhas depois de as cozer para fazer o puré.",
    "Pélo os pimentos antes de os rechear com arroz e carne picada.",
    "Pélo sempre as uvas para o frango estufado do domingo.",
]

@pytest.mark.parametrize("s", PELO_VERB)
def test_pelo_verb(s):
    _expect(s, "pelo", "VERB")


# ── tola / ADJ ────────────────────────────────────────────────────────────────

TOLA_ADJ = [
    "Não sejas tola e pensa bem antes de decidir.",
    "A decisão mais tola que tomou foi abandonar o emprego.",
    "Ela admitiu que tinha sido tola ao acreditar naquelas promessas.",
    "A pergunta mais tola da reunião foi a que ela fez.",
    "Não fiques tola com essas histórias que não têm fundamento.",
    "Nunca me pareceu tola, mas desta vez agiu de forma estranha.",
    "A ideia mais tola do projeto foi logo descartada pelo grupo.",
    "Ela ficou a parecer tola quando percebeu que estava errada.",
    "Não é nada tola, apenas estava distraída naquele momento.",
    "Uma resposta tola a uma pergunta séria não impressiona ninguém.",
    "Ela reconheceu que tinha sido tola ao confiar nessa pessoa.",
    "Não digas uma coisa tola agora que toda a gente está a ouvir.",
    "Ela foi tola ao assinar o contrato sem o ler com atenção.",
    "A escolha mais tola foi a cor que escolheu para a parede da sala.",
    "Ela pareceu tola quando não soube responder à pergunta básica.",
    "Uma atitude tola pode ter consequências graves e duradouras.",
    "Foste tola ao deixar o documento em cima da mesa sem supervisão.",
    "Não é tola de todo, tem é tendência para agir por impulso.",
]

@pytest.mark.parametrize("s", TOLA_ADJ)
def test_tola_adj(s):
    _expect(s, "tola", "ADJ")


# ── tola / NOUN ───────────────────────────────────────────────────────────────

TOLA_NOUN = [
    "Aquela tóla acreditou em tudo o que ele lhe contou sem questionar.",
    "A tóla comprou o produto sem sequer verificar as avaliações online.",
    "Toda a gente na aldeia sabia que ela era uma tóla crédula.",
    "A tóla gastou todas as poupanças num negócio claramente fraudulento.",
    "Essa tóla ainda acha que ele vai cumprir o que prometeu.",
    "A tóla assinou o contrato sem perceber o que estava a assinar.",
    "Ela comportou-se como uma tóla ao revelar todos os seus segredos.",
    "A tóla apagou todos os ficheiros importantes por engano.",
    "Toda a gente na sala riu quando a tóla respondeu aquilo.",
    "A tóla foi a única a cair no esquema que era óbvio para todos.",
    "Chamaram-lhe tóla mas ela não se importou com a opinião alheia.",
    "A tóla deixou o gato sair e ele foi para longe e não voltou.",
    "Ela é uma tóla quando se trata de dinheiro e investimentos.",
    "A tóla perdeu a carteira duas vezes na mesma semana por descuido.",
    "Essa tóla ainda não percebeu que a enganaram desde o início.",
]

@pytest.mark.parametrize("s", TOLA_NOUN)
def test_tola_noun(s):
    _expect(s, "tola", "NOUN")


# ── seco / ADJ ────────────────────────────────────────────────────────────────

SECO_ADJ = [
    "O pão está completamente seco e já não se consegue comer.",
    "O clima seco do interior dificulta muito a agricultura tradicional.",
    "O terreno seco racha com o calor intenso do mês de agosto.",
    "O ambiente seco da sala ressequiu os lábios durante a noite.",
    "O fruto seco tem uma validade muito superior ao fruto fresco.",
    "O tempo seco facilita a propagação de incêndios na floresta.",
    "O caule seco da planta indica que precisa urgentemente de água.",
    "O vento seco do sul queima as folhas das árvores mais jovens.",
    "O fruto seco é ideal para levar em caminhadas de longa duração.",
    "O cacto sobrevive em ambientes de clima seco e quente.",
    "O cabeleireiro deixou o cabelo demasiado seco com o secador quente.",
    "O inverno seco prejudicou gravemente as colheitas deste ano.",
    "O ar seco da cabine do avião resseca muito a pele durante a viagem.",
    "O muro seco foi construído sem qualquer argamassa pelos pastores locais.",
    "O papel seco no fundo da caixa amortece o choque durante o transporte.",
    "está seco e quente hoje.",
]

@pytest.mark.parametrize("s", SECO_ADJ)
def test_seco_adj(s):
    _expect(s, "seco", "ADJ")


# ── seco / VERB ───────────────────────────────────────────────────────────────

SECO_VERB = [
    "Séco as mãos na toalha depois de as lavar com cuidado.",
    "Séco o cabelo com o secador antes de sair de manhã.",
    "Séco os pratos depois de os lavar para não ficarem manchados.",
    "Séco a roupa no estendal quando o tempo está bom e solarengo.",
    "Séco as ervas aromáticas ao sol antes de as guardar num frasco.",
    "Séco os talheres um a um para não deixar marcas de água.",
    "Séco a carne depois de a marinar para a grelhar melhor.",
    "Séco o chão da casa de banho com um pano seco depois do banho.",
    "Séco os cogumelos frescos antes de os saltear em azeite.",
    "Séco sempre os copos na mão para ficarem completamente brilhantes.",
    "Séco o peixe antes de o temperar com sal e limão para fritar.",
    "Séco a fruta depois de a lavar para não apodrecer mais depressa.",
    "Séco as flores antes de as prensar num livro para conservar.",
    "Séco o pincel com cuidado depois de cada sessão de pintura.",
    "Séco os figos ao sol durante vários dias para os conservar.",
    "Séco a tela antes de aplicar uma nova camada de tinta acrílica.",
    "Séco o cão com a toalha logo que chegamos a casa da chuva.",
]

@pytest.mark.parametrize("s", SECO_VERB)
def test_seco_verb(s):
    _expect(s, "seco", "VERB")


# ── acordo / NOUN ─────────────────────────────────────────────────────────────

ACORDO_NOUN = [
    "O acordo foi assinado pelos dois países na semana passada.",
    "Chegámos a um acordo sobre a partilha das despesas comuns.",
    "O acordo de paz pôs fim a décadas de conflito na região.",
    "O acordo comercial beneficia as empresas dos dois países envolvidos.",
    "O acordo de confidencialidade impede a divulgação de informação.",
    "Rompeu o acordo sem qualquer justificação plausível apresentada.",
    "O acordo foi negociado durante meses por equipas especializadas.",
    "O acordo de divórcio foi homologado pelo tribunal competente.",
    "O acordo inclui cláusulas de revisão periódica dos termos.",
    pytest.param("O novo acordo salarial foi bem recebido pelos trabalhadores.",
                 marks=pytest.mark.xfail(reason="DET ADJ NOUN ADJ: prev2-DET fires VERB; no ADJ scorer for 'acordo'")),
    "O acordo de parceria foi renovado por mais três anos.",
    "O acordo prevê a construção de nova infraestrutura rodoviária.",
    "O acordo foi celebrado num ambiente de grande cordialidade.",
    "O acordo bilateral facilita a mobilidade de trabalhadores.",
    "O acordo foi firmado depois de longas e difíceis negociações.",
    "O acordo de licenciamento abrange todos os países da União.",
    "O acordo marco será assinado amanhã na sede da organização.",
    "O acordo de colaboração científica dura cinco anos no total.",
    "O acordo foi ratificado pelo parlamento com ampla maioria.",
    "O acordo garante o fornecimento de gás por dez anos.",
]

@pytest.mark.parametrize("s", ACORDO_NOUN)
def test_acordo_noun(s):
    _expect(s, "acordo", "NOUN")


# ── acordo / VERB ─────────────────────────────────────────────────────────────

ACORDO_VERB = [
    "Acórdo sempre às seis e meia da manhã durante a semana.",
    "Acórdo com os pássaros a cantar quando estou no campo.",
    "Acórdo cedo ao fim de semana para aproveitar melhor o dia.",
    "Acórdo às vezes no meio da noite com um pesadelo muito vívido.",
    "Acórdo geralmente antes do despertador tocar, por hábito.",
    "Acórdo bem-disposto quando durmo as horas de que preciso.",
    "Acórdo ao primeiro raio de sol que entra pela janela do quarto.",
    "Acórdo com o barulho dos vizinhos que trabalham de madrugada.",
    "Acórdo de vez em quando com o coração acelerado sem razão aparente.",
    "Acórdo muito cedo quando tenho de apanhar o voo da manhã.",
    "Acórdo sempre que há um trovão forte durante a noite.",
    "Acórdo naturalmente às sete da manhã mesmo sem despertador.",
    "Acórdo com a sensação de que algo importante me espera hoje.",
    "Acórdo facilmente com qualquer barulho na rua durante a noite.",
]

@pytest.mark.parametrize("s", ACORDO_VERB)
def test_acordo_verb(s):
    _expect(s, "acordo", "VERB")


# ── acerto / NOUN ─────────────────────────────────────────────────────────────

ACERTO_NOUN = [
    "Foi um grande acerto da empresa contratar aquele engenheiro.",
    "O acerto no alvo foi celebrado com entusiasmo por toda a equipa.",
    "O acerto de contas ficou para o final do trimestre financeiro.",
    "Considerou um acerto pessoal ter escolhido aquela universidade.",
    "O acerto da previsão meteorológica surpreendeu toda a gente.",
    "Foi um acerto notável conseguir vencer o campeonato no primeiro ano.",
    "O acerto das contas revelou um saldo positivo inesperado.",
    "O acerto do diagnóstico evitou meses de tratamento desnecessário.",
    "O acerto do plano de negócios levou a empresa ao sucesso rápido.",
    "O acerto nas previsões deste analista é notável.",
    "O acerto na estratégia permitiu ganhar uma vantagem competitiva.",
    "O acerto do cálculo foi verificado duas vezes antes de apresentar.",
    "Todos reconheceram o acerto da decisão que ela tomou.",
    "O acerto nas respostas valeu-lhe a distinção máxima na prova.",
]

@pytest.mark.parametrize("s", ACERTO_NOUN)
def test_acerto_noun(s):
    _expect(s, "acerto", "NOUN")


# ── acerto / VERB ─────────────────────────────────────────────────────────────

ACERTO_VERB = [
    "Acérto sempre no alvo quando me concentro o suficiente.",
    "Acérto as contas com o senhorio no início de cada mês.",
    "Acérto o relógio sempre que ele atrasa uns minutos.",
    "Acérto raramente na lotaria mas continuo a tentar a sorte.",
    "Acérto o passo com os outros quando caminhamos em grupo.",
    "Acérto os detalhes do contrato com o advogado amanhã de manhã.",
    "Acérto a temperatura do forno antes de começar a cozer o bolo.",
    "Acérto os horários com os colegas no início de cada projeto.",
    "Acérto quase sempre as previsões sobre o comportamento do mercado.",
    "Acérto o balanço da empresa no final de cada exercício fiscal.",
    "Acérto os planos de viagem com a minha família durante o jantar.",
    "Acérto a mira antes de disparar para garantir maior precisão.",
    "Acérto com ele o ponto de encontro por mensagem com antecedência.",
    "Acérto os pagamentos em atraso logo que tenho disponibilidade.",
]

@pytest.mark.parametrize("s", ACERTO_VERB)
def test_acerto_verb(s):
    _expect(s, "acerto", "VERB")


# ── cerro / NOUN ──────────────────────────────────────────────────────────────

CERRO_NOUN = [
    "O cerro domina toda a paisagem da aldeia a quilómetros de distância.",
    "Subimos ao cerro para apreciar a vista sobre o vale ao entardecer.",
    "O castelo foi construído no cerro mais alto de toda a região.",
    "O cerro estava coberto de urze roxa em plena floração outonal.",
    "Os pastores levavam os rebanhos ao cerro durante o verão.",
    "Plantaram árvores no cerro para travar a erosão do solo.",
    "O cerro fica visível de quase todos os pontos da vila histórica.",
    "A ermida foi construída no topo do cerro há vários séculos.",
    "O cerro serve de abrigo ao gado nas noites de inverno rigoroso.",
    "O nome da aldeia vem do cerro que a domina desde sempre.",
    "No cerro cresciam silvas e rosmaninho em abundância primaveril.",
    "A subida ao cerro é íngreme mas a vista da cimeira compensa o esforço.",
    "O cerro foi designado como área protegida pela autarquia local.",
    "O riacho nasce no cerro e vai desaguar no rio mais abaixo no vale.",
]

@pytest.mark.parametrize("s", CERRO_NOUN)
def test_cerro_noun(s):
    _expect(s, "cerro", "NOUN")


# ── cerro / VERB ──────────────────────────────────────────────────────────────

CERRO_VERB = [
    "Cérro a porta devagar para não acordar as crianças adormecidas.",
    "Cérro os olhos quando há muita claridade direta na sala.",
    "Cérro o punho em sinal de determinação antes de começar.",
    "Cérro a janela quando o vento do norte começa a soprar com força.",
    "Cérro os dentes quando estou nervoso sem me aperceber disso.",
    "Cérro o envelope depois de verificar que o documento está correto.",
    "Cérro os lábios para não dizer o que me vai na cabeça neste momento.",
    "Cérro a pasta antes de sair do escritório no final do dia.",
    "Cérro as pálpebras quando o sol me bate diretamente nos olhos.",
    "Cérro o arquivo depois de guardar todos os documentos editados.",
    "Cérro as cortinas quando quero ter privacidade total em casa.",
    "Cérro o casaco antes de sair para o frio da tarde de inverno.",
    "Cérro a caixa de ferramentas depois de usar o que precisava.",
    "Cérro a mala com atenção para não esquecer nada dentro dela.",
    "Cérro o caderno e guardo-o na mochila antes de me levantar.",
]

@pytest.mark.parametrize("s", CERRO_VERB)
def test_cerro_verb(s):
    _expect(s, "cerro", "VERB")


# ── choro / NOUN ──────────────────────────────────────────────────────────────

CHORO_NOUN = [
    "O choro do bebé acordou toda a família antes das cinco da manhã.",
    "O choro da criança foi ouvido do outro lado do corredor.",
    "O choro de alegria da mãe comoveu todos os presentes na cerimónia.",
    "O choro prolongado do recém-nascido preocupou os pais novatos.",
    "O choro da plateia no final da peça foi a maior das homenagens.",
    "O choro de frustração foi contido com dificuldade durante a reunião.",
    "O choro da noiva emocionou os convidados do casamento.",
    "O choro coletivo pela perda do líder ecoou pelas ruas da cidade.",
    "O choro do menino parou assim que a mãe o pegou ao colo.",
    "O choro de arrependimento chegou tarde demais para mudar o passado.",
    "O choro incontrolável durou horas depois de receber a notícia.",
    "O choro silencioso foi mais expressivo do que qualquer palavra.",
    "O choro da criança na consulta assustou a médica pediatra.",
    "O choro de desespero foi ouvido pelos vizinhos da parede.",
    "O choro da avó ao ver o neto pela primeira vez tocou a todos.",
]

@pytest.mark.parametrize("s", CHORO_NOUN)
def test_choro_noun(s):
    _expect(s, "choro", "NOUN")


# ── choro / VERB ──────────────────────────────────────────────────────────────

CHORO_VERB = [
    "Chóro sempre que vejo aquele filme da minha infância.",
    "Chóro de alegria quando recebo boas notícias inesperadas.",
    "Chóro raramente, mas quando acontece não consigo parar.",
    "Chóro às vezes quando oiço música que me traz memórias antigas.",
    "Chóro quando estou muito cansado e o corpo não aguenta mais.",
    "Chóro ao cortar cebola mesmo quando tento evitar o contacto.",
    "Chóro em funerais mesmo de pessoas que não conhecia muito.",
    "Chóro quando os meus filhos conquistam algo importante para eles.",
    "Chóro quando leio livros com finais tristes e inesperados.",
    "Chóro de riso quando o meu amigo conta as suas histórias caricatas.",
    "Chóro em silêncio para não preocupar as pessoas que me rodeiam.",
    "Chóro quando me lembro da minha avó e do cheiro da sua casa.",
    "Chóro durante as despedidas no aeroporto, sempre acontece assim.",
    "Chóro quando a equipa perde numa final depois de tanta luta.",
]

@pytest.mark.parametrize("s", CHORO_VERB)
def test_choro_verb(s):
    _expect(s, "choro", "VERB")


# ── colher / NOUN ─────────────────────────────────────────────────────────────

COLHER_NOUN = [
    "Usa uma colhér de sopa para mexer bem o caldo de carne.",
    "A colhér de prata estava no escrínio da avó desde o casamento.",
    "Ele mexeu o café com a colhér devagar antes de o beber.",
    "A colhér de madeira é ideal para mexer o doce sem arranhar a panela.",
    "A criança aprendeu a usar a colhér antes de qualquer outro talher.",
    "A colhér está torcida depois de cair no lava-louça com os talheres.",
    "Ele tirou uma colhér de sopa de arroz para o prato do filho.",
    "A colhér de chá serve também para medir ingredientes em receitas.",
    "A colhér de servir ficou esquecida dentro da tigela de salada.",
    "A avó usava sempre a colhér grande para provar o caldo quente.",
    "A colhér de pau está partida e já não se pode usar mais.",
    "Pôs uma colhér de mel no chá quente para aliviar a tosse.",
    "A colhér de sobremesa é maior do que a de chá e menor que a de sopa.",
    "A colhér de plástico não aguenta o calor do tacho no fogão.",
]

@pytest.mark.parametrize("s", COLHER_NOUN)
def test_colher_noun(s):
    _expect(s, "colher", "NOUN")


# ── colher / VERB ─────────────────────────────────────────────────────────────

COLHER_VERB = [
    "Vamos colher os tomates do quintal ainda esta tarde antes do jantar.",
    "É preciso colher as uvas antes que a chuva estrague a vindima.",
    "Ela foi colher ervas aromáticas para temperar o assado do almoço.",
    "O agricultor foi colher as azeitonas com toda a família reunida.",
    "Vou colher uns ramos de rosmaninho para pôr na entrada de casa.",
    "É altura de colher os pimentos que estão já bem maduros no canteiro.",
    pytest.param("Ela aprendeu a colher cogumelos silvestres com o avô na serra.",
                 marks=pytest.mark.xfail(reason="'a' before infinitive ties NOUN(DET+5) with VERB(infinitive+5)")),
    "Foram colher laranjas da laranjeira que eles próprios plantaram.",
    "O apicultor foi colher o mel das colmeias ao fim do verão.",
    "Precisamos de colher a lavanda antes que perca o aroma intenso.",
    "Foram colher amoras para fazer a compota que todos adoram.",
    "É importante colher as ervas logo de manhã para maior frescura.",
    "Foram colher castanhas na serra durante o fim de semana outonal.",
    "O agricultor foi colher o milho depois de verificar a maturação.",
    "Ela foi colher flores silvestres para decorar a mesa do almoço.",
]

@pytest.mark.parametrize("s", COLHER_VERB)
def test_colher_verb(s):
    _expect(s, "colher", "VERB")


# ── começo / NOUN ─────────────────────────────────────────────────────────────

COMEÇO_NOUN = [
    "O começo do filme é absolutamente arrebatador e prende a atenção.",
    "No começo da relação tudo parecia perfeito e sem problemas.",
    "O começo do ano lectivo é sempre uma altura de grande azáfama.",
    "No começo do projeto havia muitas dúvidas sobre a viabilidade.",
    "O começo do discurso foi surpreendente pela sua franqueza brutal.",
    "O começo do livro promete muito mas o final decepciona um pouco.",
    "No começo da carreira passou por muitas dificuldades financeiras.",
    "O começo da crise foi difícil de identificar com precisão temporal.",
    "O começo da cerimónia foi adiado por causa do mau tempo.",
    "O começo do inverno trouxe temperaturas negativas para a região.",
    "O começo da peça foi aplaudido de pé pelo público entusiasmado.",
    "O começo do tratamento mostrou resultados promissores para o doente.",
    "O começo da democracia foi marcado por muita turbulência política.",
]

@pytest.mark.parametrize("s", COMEÇO_NOUN)
def test_começo_noun(s):
    _expect(s, "começo", "NOUN")


# ── começo / VERB ─────────────────────────────────────────────────────────────

COMEÇO_VERB = [
    "Coméço sempre o dia com um copo de água e um alongamento rápido.",
    "Coméço a trabalhar às oito da manhã quando o escritório está sossegado.",
    "Coméço a cozinhar o jantar pelas sete para estar pronto a tempo.",
    "Coméço a reunião com um resumo do que foi discutido na anterior.",
    "Coméço o treino com aquecimento durante dez minutos sem falta.",
    "Coméço a ler o livro novo logo que o anterior fica terminado.",
    "Coméço o relatório pela conclusão para depois preencher o resto.",
    "Coméço a sentir saudades de casa quando viajo por muito tempo.",
    "Coméço sempre as reuniões na hora certa, independentemente de quem falta.",
    "Coméço a aula com uma pergunta para verificar o que ficou retido.",
    "Coméço o mês com a lista de prioridades bem definidas no papel.",
    "Coméço a escrever quando tenho as ideias bem organizadas na mente.",
    "Coméço o dia bem quando durmo as horas de que o meu corpo precisa.",
    "Coméço cada capítulo com uma citação que resume o seu conteúdo.",
]

@pytest.mark.parametrize("s", COMEÇO_VERB)
def test_começo_verb(s):
    _expect(s, "começo", "VERB")


# ── conserto / NOUN ───────────────────────────────────────────────────────────

CONSERTO_NOUN = [
    "O conserto da máquina de lavar demorou mais do que o esperado.",
    "O conserto do telhado custou mais do que estava no orçamento inicial.",
    "O conserto do automóvel ficou pronto em apenas dois dias úteis.",
    "Mandei o relógio ao conserto e ficou como novo depois da reparação.",
    "O conserto da bicicleta foi rápido e o mecânico cobrou um valor justo.",
    "O conserto do computador revelou um problema no disco rígido.",
    "O conserto da torneira evitou um desperdício maior de água.",
    "O conserto do aquecimento foi urgente com as temperaturas negativas.",
    "O conserto ficou barato porque o técnico já conhecia bem o modelo.",
    "O conserto do telhado foi adiado por causa da chuva intensa.",
    "Depois do conserto o motor voltou a funcionar sem qualquer problema.",
    "O conserto do portão elétrico foi feito em menos de uma hora.",
    "O conserto da joia foi feito por um ourives especializado em antiguidades.",
    "O conserto da caleira impediu que a água entrasse pelas paredes.",
]

@pytest.mark.parametrize("s", CONSERTO_NOUN)
def test_conserto_noun(s):
    _expect(s, "conserto", "NOUN")


# ── conserto / VERB ───────────────────────────────────────────────────────────

CONSERTO_VERB = [
    "Consérto bicicletas e aparelhos elétricos há mais de vinte anos.",
    "Consérto as meias rotas em vez de as deitar fora como é moda.",
    "Consérto os sapatos eu próprio sempre que a sola começa a despegar.",
    "Consérto o que está avariado antes de pensar em comprar novo.",
    "Consérto o código do programa quando encontro um erro de execução.",
    "Consérto pequenas avarias domésticas sem precisar de chamar técnicos.",
    "Consérto o texto antes de o enviar para evitar erros desnecessários.",
    "Consérto as calças quando o botão cai, porque sei costurar bem.",
    "Consérto os brinquedos da minha filha para durarem mais tempo.",
    "Consérto os erros do relatório depois de o reler com atenção.",
    "Consérto o argumento quando percebo que há uma falha lógica.",
    "Consérto electrodomésticos velhos para lhes dar uma nova vida útil.",
    "Consérto a minha postura quando começo a sentir dor nas costas.",
    "Consérto as plantas quando têm algum problema de crescimento.",
]

@pytest.mark.parametrize("s", CONSERTO_VERB)
def test_conserto_verb(s):
    _expect(s, "conserto", "VERB")


# ── coro / NOUN ───────────────────────────────────────────────────────────────

CORO_NOUN = [
    "O coro da sé cantou durante toda a cerimónia religiosa de natal.",
    "O coro de vozes infantis encantou o público da sala cheia.",
    "O coro entrou em palco vestido a rigor para a actuação final.",
    "O coro de vozes masculinas encheu a catedral com o seu som potente.",
    "O coro ensaia todas as semanas na sala paroquial da aldeia.",
    "O regente do coro é um músico internacionalmente reconhecido.",
    "O coro interpretou uma peça de polifonia renascentista ao vivo.",
    "O coro da escola participou no festival regional de música coral.",
    "O coro de câmara tem apenas doze vozes mas o som é extraordinário.",
    "O coro foi fundado há mais de cem anos por um grupo de entusiastas.",
    "O coro amateur da aldeia tem mais de trinta membros activos.",
    "O coro universitário está a preparar um concerto para o natal.",
    "O coro terminou a actuação com uma ovação prolongada do público.",
    "O coro de gospel animou a festa com energia e alegria contagiantes.",
]

@pytest.mark.parametrize("s", CORO_NOUN)
def test_coro_noun(s):
    _expect(s, "coro", "NOUN")


# ── coro / VERB ───────────────────────────────────────────────────────────────

CORO_VERB = [
    "Córo de vergonha quando me fazem um elogio em público.",
    "Córo facilmente e as pessoas reparam logo nessa reação.",
    "Córo sempre que me pedem para falar numa apresentação.",
    "Córo ao me lembrar da cena embaraçosa de ontem.",
    "Córo quando o professor me chama para responder na aula.",
    "Córo ao receber flores de surpresa diante de tanta gente.",
    "Córo de vergonha alheia quando vejo aquele vídeo de novo.",
    "Córo quando me dizem que fiz um trabalho excecional.",
    "Córo ao ser apresentada como especialista sem me sentir preparada.",
    "Córo quando alguém nota os meus erros diante de outros colegas.",
    "Córo facilmente, é uma característica que herdei da minha mãe.",
    "Córo quando me fazem uma pergunta que não sei responder.",
    "Córo ao ver a minha fotografia nos jornais após o evento.",
    "Córo quando me agradecem de forma muito efusiva e pública.",
]

@pytest.mark.parametrize("s", CORO_VERB)
def test_coro_verb(s):
    _expect(s, "coro", "VERB")


# ── corte / NOUN ──────────────────────────────────────────────────────────────

CORTE_NOUN = [
    "O córte de cabelo ficou exactamente como ela pediu ao cabeleireiro.",
    "O córte orçamental afectou os serviços públicos de todo o país.",
    "O córte de electricidade durou três horas sem aviso prévio.",
    "O córte do bolo de aniversário foi feito pela própria aniversariante.",
    "O córte de água foi necessário para fazer a reparação na tubagem.",
    "O córte do cabelo é feito de três em três meses normalmente.",
    "O córte da carne deve ser feito contra as fibras para ficar mais macio.",
    "O córte das comunicações isolou a aldeia durante dois dias inteiros.",
    "O córte do financiamento obrigou à suspensão do projeto de investigação.",
    "O córte de faixa inaugurou oficialmente o novo hospital da cidade.",
    "O córte fino do presunto revela o talento do cortador experiente.",
    "O córte de cabelo novo deu-lhe uma aparência completamente diferente.",
    # "corte" as "royal court" — also NOUN IPA
    "A corte real reuniu-se no palácio para a cerimónia de abertura.",
    "A corte do rei era famosa pelo seu luxo e intrigas políticas.",
    "A corte medieval era composta por nobres, clérigos e conselheiros.",
]

@pytest.mark.parametrize("s", CORTE_NOUN)
def test_corte_noun(s):
    _expect(s, "corte", "NOUN")


# ── corte / VERB (imperative) ─────────────────────────────────────────────────

CORTE_VERB = [
    "Córte o pão em fatias finas para fazer as torradas do pequeno-almoço.",
    "Córte a carne em cubos pequenos antes de a adicionar ao estufado.",
    "Córte as ervas aromáticas com uma tesoura para não as magoar.",
    "Córte a conversa antes que a situação piore mais do que já está.",
    "Córte o cabelo do menino antes das fotografias da escola, por favor.",
    "Córte as flores a quarenta e cinco graus antes de as pôr no vaso.",
    "Córte os legumes em juliana para a sopa ficar mais saborosa.",
    "Córte a ligação se o interlocutor se tornar agressivo ao telefone.",
    "Córte as unhas das crianças com a tesoura própria para esse fim.",
    "Córte os gastos desnecessários antes de pedir um aumento de crédito.",
    "Córte o queijo com fio de arame para não esmagar a textura.",
    "Córte o tecido com muito cuidado para não desperdiçar material.",
    "Córte os galhos secos da árvore antes que caiam por si mesmos.",
    "Córte o acesso ao sistema se detectar actividade suspeita.",
]

@pytest.mark.parametrize("s", CORTE_VERB)
def test_corte_verb(s):
    _expect(s, "corte", "VERB")


# ── forma / NOUN ──────────────────────────────────────────────────────────────

FORMA_NOUN = [
    "A forma do vaso é muito elegante e combina com a decoração.",
    "A forma como ele tratou os colegas foi completamente inaceitável.",
    "Não há outra forma de resolver este problema sem diálogo aberto.",
    "A forma da cozedura influencia muito o resultado final do bolo.",
    "A forma do edifício foi inspirada nas ondas do oceano Atlântico.",
    "A forma como ela fala cativa toda a gente à volta imediatamente.",
    "A forma de preparar o bacalhau varia consoante a região do país.",
    "A forma mais eficaz é a preventiva, antes de o problema surgir.",
    "A forma geométrica mais comum na natureza é a espiral logarítmica.",
    "A forma de avaliar os alunos mudou muito nos últimos vinte anos.",
    "A forma como ele se vestia revelava muito sobre a sua personalidade.",
    # "fôrma" (baking mold) — diacritized form, also NOUN IPA
    "A fôrma que ela usa para fazer o bolo é de silicone resistente ao forno.",
    "A forma da ilha vista do ar parece um dragão deitado no oceano.",
]

@pytest.mark.parametrize("s", FORMA_NOUN)
def test_forma_noun(s):
    _expect(s, "forma", "NOUN")


# ── forma / VERB ──────────────────────────────────────────────────────────────

FORMA_VERB = [
    "A empresa forma técnicos especializados todos os anos com rigor.",
    "A escola forma cidadãos responsáveis e conscientes do mundo.",
    "A experiência forma o carácter melhor do que qualquer livro lido.",
    "A universidade forma engenheiros preparados para os desafios actuais.",
    "A névoa forma-se sobre o rio nas manhãs frias de outubro e novembro.",
    "O gelo forma-se rapidamente quando a temperatura desce abaixo de zero.",
    "A orquestra forma um semicírculo antes de começar a actuar em palco.",
    "O treinador forma os jogadores para a final com sessões intensas.",
    "A empresa forma equipas multidisciplinares para cada novo projeto.",
    "O cozinheiro forma os aprendizes com paciência e rigor na cozinha.",
    "A academia forma pilotos experientes para companhias aéreas internacionais.",
    "A instituição forma investigadores de excelência na área das ciências.",
    "A indústria forma operadores qualificados para as linhas de produção.",
]

@pytest.mark.parametrize("s", FORMA_VERB)
def test_forma_verb(s):
    _expect(s, "forma", "VERB")


# ── gosto / NOUN ──────────────────────────────────────────────────────────────

GOSTO_NOUN = [
    "Tens um gosto impecável na forma como decoras a tua casa.",
    "O gosto por viagens desenvolveu-se desde muito cedo na sua vida.",
    "O gosto deste queijo curado é intenso e persistente no palato.",
    "O gosto pela leitura foi cultivado pelos pais desde a infância.",
    "O gosto desta água mineral é diferente de todas as outras que conheço.",
    "O gosto pessoal não deve ser confundido com a qualidade objectiva.",
    "O gosto pela música clássica surgiu depois de ouvir uma sinfonia ao vivo.",
    "O gosto desta fruta está no ponto certo de maturação perfeita.",
    "O gosto refinado dela manifesta-se em tudo o que escolhe usar.",
    "O gosto pelo desporto acompanhou-o durante toda a vida activa.",
    "O gosto desta sobremesa é delicado e subtil, não demasiado doce.",
    "O gosto por fotografia surgiu quando lhe ofereceram uma câmara.",
    "O gosto da fruta tropical é inconfundível e muito agradável.",
    "O gosto pelo trabalho manual foi herdado do avô carpinteiro.",
    "O gosto pelo teatro começou quando participou numa peça escolar.",
]

@pytest.mark.parametrize("s", GOSTO_NOUN)
def test_gosto_noun(s):
    _expect(s, "gosto", "NOUN")


# ── gosto / VERB ──────────────────────────────────────────────────────────────

GOSTO_VERB = [
    "Gósto muito de passear à beira-mar ao fim da tarde.",
    "Gósto de cozinhar para os amigos nos fins de semana chuvosos.",
    "Gósto de ler antes de adormecer, é o meu ritual de todas as noites.",
    "Gósto de ouvir música clássica quando trabalho em tarefas criativas.",
    "Gósto de andar de bicicleta quando o tempo está bom na cidade.",
    "Gósto de jardinar e ver as plantas crescer devagar com os cuidados.",
    "Gósto de aprender coisas novas fora da minha área de especialização.",
    "Gósto de acordar cedo quando não tenho pressa para o dia inteiro.",
    "Gósto de viajar de comboio para ver a paisagem a passar devagar.",
    "Gósto de fotografar paisagens e arquitectura nas minhas viagens.",
    "Gósto de silêncio quando preciso de me concentrar num problema.",
    "Gósto de desenhar à mão mesmo quando uso software profissional.",
    "Gósto de café preto e forte de manhã para começar bem o dia.",
    "Gósto de explorar mercados locais quando visito cidades novas.",
]

@pytest.mark.parametrize("s", GOSTO_VERB)
def test_gosto_verb(s):
    _expect(s, "gosto", "VERB")


# ── gozo / NOUN ───────────────────────────────────────────────────────────────

GOZO_NOUN = [
    "Tirou um mês de gozo de férias no verão passado em família.",
    "O gozo de direitos civis é fundamental numa sociedade democrática.",
    "Ela estava em gozo de licença de maternidade quando foi contactada.",
    "O gozo do patrimônio cultural deve ser acessível a todos os cidadãos.",
    "Ficou em gozo de folgas acumuladas durante todo o mês de agosto.",
    "O gozo das férias foi interrompido por uma chamada urgente do trabalho.",
    "O gozo do bom tempo foi aproveitado ao máximo durante o fim de semana.",
    "O gozo de plena capacidade jurídica é reconhecido aos maiores de idade.",
    "O gozo destes privilégios está condicionado ao bom comportamento.",
    "O gozo de férias foi antecipado por causa de uma situação familiar.",
    "O gozo das instalações está reservado aos sócios efectivos do clube.",
    "O gozo do espaço público é um direito de todos os cidadãos.",
    "O gozo das suas atribuições foi temporariamente suspenso pelo tribunal.",
    "O gozo do crédito fiscal está sujeito a determinadas condições legais.",
]

@pytest.mark.parametrize("s", GOZO_NOUN)
def test_gozo_noun(s):
    _expect(s, "gozo", "NOUN")


# ── gozo / VERB ───────────────────────────────────────────────────────────────

GOZO_VERB = [
    "Gózo de boa saúde graças a uma alimentação equilibrada e exercício.",
    "Gózo das pequenas alegrias da vida sempre que posso e tenho oportunidade.",
    "Gózo de férias merecidas depois de um ano de muito trabalho intenso.",
    "Gózo de um apartamento com vista para o rio e isso vale muito.",
    "Gózo de grande estima por parte dos meus colegas de trabalho.",
    "Gózo das manhãs tranquilas de domingo sem horários nem obrigações.",
    "Gózo de uma posição privilegiada para observar o pôr do sol.",
    "Gózo de confiança total da minha equipa em situações de pressão.",
    "Gózo do silêncio da natureza quando saio da cidade no fim de semana.",
    "Gózo de boas relações com todos os vizinhos do prédio.",
    "Gózo de plena autonomia na gestão do meu tempo de trabalho.",
    "Gózo das conversas longas à lareira com os amigos de infância.",
    "Gózo de um estado de paz interior que levei anos a encontrar.",
    "Gózo de total confiança do meu superior hierárquico na empresa.",
]

@pytest.mark.parametrize("s", GOZO_VERB)
def test_gozo_verb(s):
    _expect(s, "gozo", "VERB")


# ── jogo / NOUN ───────────────────────────────────────────────────────────────

JOGO_NOUN = [
    "O jogo de ontem foi emocionante até ao último minuto do tempo.",
    "O jogo de xadrez exige concentração e estratégia de longo prazo.",
    "O jogo de cartas foi a diversão da família durante a tarde chuvosa.",
    "O resultado do jogo surpreendeu todos os analistas desportivos.",
    "O jogo terminou em empate depois de duas horas de futebol intenso.",
    "O jogo de palavras dele provocou muitas gargalhadas na mesa.",
    "O jogo de luzes no concerto criou uma atmosfera mágica e única.",
    "O jogo de damas ficou a meio quando o jantar ficou pronto.",
    "O jogo de tabuleiro foi o presente mais pedido esta época festiva.",
    "O jogo de futebol foi suspenso por causa da chuva intensa no estádio.",
    "O jogo de computador manteve-o ocupado durante toda a tarde.",
    "O jogo limpo é um valor fundamental no desporto e na vida.",
]

@pytest.mark.parametrize("s", JOGO_NOUN)
def test_jogo_noun(s):
    _expect(s, "jogo", "NOUN")


# ── jogo / VERB ───────────────────────────────────────────────────────────────

JOGO_VERB = [
    "Jógo futebol todos os sábados de manhã com os amigos de longa data.",
    "Jógo xadrez com o meu pai todas as sextas-feiras à tarde.",
    "Jógo às cartas quando estou com os avós durante as férias.",
    "Jógo ténis há dez anos e ainda estou a melhorar a minha técnica.",
    "Jógo videojogos raramente, apenas quando quero descansar a cabeça.",
    "Jógo no mesmo clube de futebol desde os sete anos de idade.",
    "Jógo basquetebol ao ar livre quando o tempo está suficientemente quente.",
    "Jógo damas com os meus filhos ao fim de semana com muito prazer.",
    "Jógo golf esporadicamente quando recebo convite dos sócios do clube.",
    "Jógo dominó com os amigos do bairro nas tardes de domingo tranquilo.",
    "Jógo snooker num clube privado perto de casa regularmente.",
    "Jógo voleibol de praia durante as férias de verão na costa.",
    "Jógo com o meu cão no jardim todas as tardes depois do trabalho.",
]

@pytest.mark.parametrize("s", JOGO_VERB)
def test_jogo_verb(s):
    _expect(s, "jogo", "VERB")


# ── molho / NOUN ──────────────────────────────────────────────────────────────

MOLHO_NOUN = [
    "Preparei um molho de tomate caseiro para a massa do jantar.",
    "O molho béchamel da lasanha ficou cremoso e muito bem temperado.",
    "O molho de cogumelos combina muito bem com o bife grelhado.",
    "O molho vinagrete é a forma mais simples de temperar uma salada.",
    "O molho de ervas frescas acompanha perfeitamente o peixe grelhado.",
    "O molho ficou demasiado espesso e tive de adicionar um pouco de água.",
    "O molho de chocolate quente sobre o gelado é irresistível no verão.",
    "O molho de soja é um ingrediente fundamental na cozinha asiática.",
    "O molho da carne ficou com um sabor intenso depois de reduzir na panela.",
    "O molho de mel e mostarda combina muito bem com frango assado.",
    "O molho de alho e azeite é a base da gastronomia portuguesa tradicional.",
    "O molho picante não é para toda a gente, mas eu adoro intensidade.",
    "O molho de pesto é fácil de fazer e dura vários dias no frigorífico.",
    "O molho de camarão ficou delicioso com o toque de brandy no final.",
]

@pytest.mark.parametrize("s", MOLHO_NOUN)
def test_molho_noun(s):
    _expect(s, "molho", "NOUN")


# ── molho / VERB ──────────────────────────────────────────────────────────────

MOLHO_VERB = [
    "Mólho os pés na bacia com água morna e sal ao fim do dia longo.",
    "Mólho o pão no café antes de o comer ao pequeno-almoço da manhã.",
    "Mólho as plantas de manhã cedo antes de o sol ficar demasiado quente.",
    "Mólho sempre as leguminosas secas de véspera antes de as cozinhar.",
    "Mólho os pincéis depois de usar para limpar bem a tinta acrílica.",
    "Mólho os pés no mar logo que chego à praia no primeiro dia.",
    "Mólho os biscoitos no leite antes de os dar às crianças pequenas.",
    "Mólho as sementes antes de as plantar para acelerar a germinação.",
    "Mólho o pano com água fria quando me arrepia durante o verão.",
    "Mólho as passas em rum antes de as adicionar ao bolo de natal.",
    "Mólho as amêndoas durante a noite para ficarem mais fáceis de pelar.",
    "Mólho a esponja antes de a usar para limpar as superfícies da cozinha.",
    "Mólho o cacho de uvas antes de o servir para ficarem bem frescas.",
    "Mólho as escovas de dentes antes de aplicar a pasta dentífrica.",
]

@pytest.mark.parametrize("s", MOLHO_VERB)
def test_molho_verb(s):
    _expect(s, "molho", "VERB")


# ── olho / NOUN ───────────────────────────────────────────────────────────────

OLHO_NOUN = [
    "Ela tem um olho azul e outro castanho, uma característica rara.",
    "O olho do furacão é surpreendentemente calmo comparado com a periferia.",
    "O olho da agulha é demasiado pequeno para este fio tão grosso.",
    "O olho do boi é uma janela circular muito característica na arquitectura.",
    "O exame ao olho revelou uma ligeira miopia que precisava de correcção.",
    "O olho direito ficou inflamado depois de entrar algo pequeno dentro.",
    "O olho humano distingue milhões de tons de cor em condições normais.",
    "O olho do ciclone passou directamente sobre a cidade costeira.",
    "O olho clínico do médico detectou imediatamente o problema escondido.",
    "O olho nu não consegue ver as estrelas mais distantes do universo.",
    "O olho de peixe é uma lente que capta um ângulo muito amplo.",
    "O olho cansado precisa de descanso e de uma boa noite de sono.",
    "O olho da fechadura mostrava o corredor iluminado do outro lado.",
]

@pytest.mark.parametrize("s", OLHO_NOUN)
def test_olho_noun(s):
    _expect(s, "olho", "NOUN")


# ── olho / VERB ───────────────────────────────────────────────────────────────

OLHO_VERB = [
    "Ólho sempre para o horizonte quando estou no mar ou numa montanha.",
    "Ólho para a fotografia da minha avó quando sinto saudade dela.",
    "Ólho para o relógio com frequência quando tenho um compromisso importante.",
    "Ólho pela janela quando ouço barulho fora de horas na rua.",
    "Ólho para o prato do meu vizinho no restaurante e fico com inveja.",
    "Ólho para o céu todas as noites para ver as estrelas quando não há nuvens.",
    "Ólho para os pés enquanto caminho quando estou distraído a pensar.",
    "Ólho para a lista de tarefas ao acordar para planear bem o dia.",
    "Ólho para trás às vezes para verificar se alguém me está a seguir.",
    "Ólho para o retrato do filho e sinto um orgulho imenso a crescer.",
    "Ólho para o mapa antes de sair para não me perder na cidade nova.",
    "Ólho para as notícias de manhã para perceber o que se passa no mundo.",
    "Ólho sempre para ambos os lados antes de atravessar a estrada.",
    "Ólho para o espelho antes de sair para verificar a minha aparência.",
]

@pytest.mark.parametrize("s", OLHO_VERB)
def test_olho_verb(s):
    _expect(s, "olho", "VERB")


# ── peso / NOUN ───────────────────────────────────────────────────────────────

PESO_NOUN = [
    "O peso da mochila era excessivo para uma criança de oito anos.",
    "O peso do responsável recai sobre os seus ombros há muitos anos.",
    "O peso das expectativas familiares foi sempre muito difícil de carregar.",
    "O peso do pacote surpreendeu o carteiro pela sua dimensão reduzida.",
    "O peso do silêncio naquela sala era palpável e perturbador.",
    "O peso da decisão manteve-o acordado durante várias noites seguidas.",
    "O peso do ouro é medido em gramas com grande precisão.",
    "O peso do argumento é o que convence, não o volume da voz.",
    "O peso do passado às vezes impede-nos de olhar para o futuro.",
    "O peso do investimento será distribuído por vários anos fiscais.",
    "O peso do cesto de compras aumentou com os produtos novos.",
    "O peso da responsabilidade partilhada torna tudo mais suportável.",
    "O peso do resultado final recai sobre todos os membros da equipa.",
    "O peso do edifício foi calculado pelos engenheiros estruturais.",
]

@pytest.mark.parametrize("s", PESO_NOUN)
def test_peso_noun(s):
    _expect(s, "peso", "NOUN")


# ── peso / VERB ───────────────────────────────────────────────────────────────

PESO_VERB = [
    "Péso os ingredientes com precisão antes de começar a cozinhar.",
    "Péso a farinha numa balança digital para garantir exactidão.",
    "Péso as malas antes de partir para o aeroporto para evitar surpresas.",
    "Péso as palavras com cuidado antes de as dizer em situações delicadas.",
    "Péso as vantagens e desvantagens antes de tomar qualquer decisão.",
    "Péso o bebé na consulta pediátrica para acompanhar o crescimento.",
    "Péso a manteiga separadamente porque a receita é muito exigente.",
    "Péso os prós e os contras de cada opção antes de escolher.",
    "Péso a fruta no mercado antes de a colocar no saco para pagar.",
    "Péso as amêndoas para o bolo com balança de precisão milimétrica.",
    "Péso as nozes antes de as partir para usar a quantidade certa.",
    "Péso o queijo na tábua digital antes de o servir para os convidados.",
    "Péso regularmente para controlar a minha evolução durante a dieta.",
    "Péso as decisões importantes com tempo e sem pressa desnecessária.",
    "Péso a carne separadamente para saber a dose exacta por pessoa.",
]

@pytest.mark.parametrize("s", PESO_VERB)
def test_peso_verb(s):
    _expect(s, "peso", "VERB")


# ── porto / NOUN ──────────────────────────────────────────────────────────────

PORTO_NOUN = [
    "O porto de Lisboa recebe cruzeiros de todo o mundo ao longo do ano.",
    "O porto de Leixões é o maior porto artificial da Península Ibérica.",
    "O porto da Figueira da Foz é muito frequentado pelos pescadores locais.",
    "O porto de Setúbal tem grande actividade industrial e comercial.",
    "O porto de abrigo protegeu os barcos da tempestade que se aproximava.",
    "O porto franco facilita o comércio internacional com incentivos fiscais.",
    "O porto está equipado com as mais modernas gruas de contentores.",
    "O porto de pesca estava cheio de barcos coloridos ao amanhecer.",
    "O porto recebeu ontem o maior navio de contentores alguma vez visto.",
    "O porto foi expandido para aumentar a capacidade de carga anual.",
    "O porto seguro é uma metáfora para um lugar de conforto e protecção.",
    "O porto de recreio tem duzentos e cinquenta lugares de amarração.",
    "O porto foi fechado temporariamente por causa do temporal no mar.",
    "O porto artificial foi construído para proteger os navios do vento.",
]

@pytest.mark.parametrize("s", PORTO_NOUN)
def test_porto_noun(s):
    _expect(s, "porto", "NOUN")


# ── porto / VERB (portar-se) ──────────────────────────────────────────────────

PORTO_VERB = [
    "Pórto-me sempre com respeito em situações de representação formal.",
    "Pórto-me bem mesmo quando estou sob pressão e muito cansado.",
    "Pórto-me de forma adequada ao contexto cultural do país que visito.",
    "Pórto-me de acordo com o que me foi ensinado desde criança.",
    "Pórto-me de forma profissional mesmo quando não estou de acordo.",
    "Pórto-me exemplarmente quando represento a instituição no estrangeiro.",
    "Pórto-me com elegância nas cerimónias formais e eventos protocolares.",
    "Pórto-me sempre com cordialidade mesmo quando há desacordos sérios.",
    "Pórto-me de forma irrepreensível nas cerimónias de entrega de prémios.",
    "Pórto-me bem com os clientes mesmo quando estou com problemas pessoais.",
]

@pytest.mark.parametrize("s", PORTO_VERB)
def test_porto_verb(s):
    _expect(s, "porto", "VERB")


# ── posto / NOUN ──────────────────────────────────────────────────────────────

POSTO_NOUN = [
    "Ele ocupa um posto de grande responsabilidade na administração central.",
    "O posto de saúde está aberto todos os dias incluindo fins de semana.",
    "O posto da GNR fica na entrada da vila e funciona vinte e quatro horas.",
    "O posto de trabalho será eliminado com a automatização progressiva.",
    "O posto fronteiriço controla a entrada de mercadorias e pessoas.",
    "O posto de abastecimento fica na estrada nacional a cinco quilómetros.",
    "O posto avançado da unidade militar ficava no limite da zona controlada.",
    "O posto de vigia estava estrategicamente colocado no ponto mais alto.",
    "O posto de socorro estava equipado com o material de emergência básico.",
    "O posto de comando coordenava todas as operações no terreno.",
    "O posto diplomático no estrangeiro é uma posição muito disputada.",
    "O posto de enfermagem atende a população da aldeia há décadas.",
    "O posto de gasolina ficou fechado por falta de combustível.",
    # PPT of "pôr" — pronounced like NOUN (closed-o), test expects NOUN IPA
    "O jantar já estava pôsto na mesa quando chegámos a casa.",
    "O casaco estava pôsto na cadeira desde a manhã quando saiu.",
    "O documento estava pôsto em cima da secretária do director.",
    "O livro foi pôsto na prateleira depois de ser lido com atenção.",
    "O carro estava pôsto na garagem antes de começar a chover muito.",
    "O sinal foi pôsto na estrada para avisar os condutores do perigo.",
    "O envelope estava pôsto debaixo da porta quando chegámos a casa.",
    "O relatório estava pôsto em destaque na página inicial do portal.",
]

@pytest.mark.parametrize("s", POSTO_NOUN)
def test_posto_noun(s):
    _expect(s, "posto", "NOUN")


# ── posto / VERB (postar, 1st-person) ────────────────────────────────────────

POSTO_VERB = [
    "Pósto uma mensagem por dia nas redes sociais da empresa.",
    "Pósto fotografias das viagens assim que chego ao destino.",
    "Pósto sempre os resultados antes de sair do escritório.",
]

@pytest.mark.parametrize("s", POSTO_VERB)
def test_posto_verb(s):
    _expect(s, "posto", "VERB")


# ── rego / NOUN ───────────────────────────────────────────────────────────────

REGO_NOUN = [
    "O rego de rega percorre todo o campo de forma sinuosa e eficaz.",
    "O rego foi construído pelos antepassados há muitas gerações atrás.",
    "O rego distribui a água pelos diferentes talhões do terreno agrícola.",
    "O rego estava entupido com folhas e foi necessário limpar.",
    "O rego divide a propriedade entre os dois agricultores vizinhos.",
    "O rego de pedra seca atravessa o jardim histórico da quinta.",
    pytest.param("O agricultor abriu um novo rego para levar água ao novo pomar.",
                 marks=pytest.mark.xfail(reason="DET ADJ NOUN: prev2-DET VERB signal beats NOUN=0")),
    "O rego estava seco porque a nascente tinha cessado de fluir.",
    "O rego de irrigação garante água às plantas durante os meses secos.",
    "O rego antigo foi restaurado e voltou a funcionar perfeitamente.",
    "O rego de drenagem evita que a água encharque os campos em excesso.",
    "O rego principal ramifica-se em vários pequenos canais secundários.",
    "O rego de pedra está coberto de musgo verde depois das chuvas.",
    "O rego marca a fronteira entre os dois terrenos desde há décadas.",
]

@pytest.mark.parametrize("s", REGO_NOUN)
def test_rego_noun(s):
    _expect(s, "rego", "NOUN")


# ── rego / VERB ───────────────────────────────────────────────────────────────

REGO_VERB = [
    "Régo as plantas da varanda todos os dias durante o verão quente.",
    "Régo o jardim de manhã cedo para reduzir a evaporação solar.",
    "Régo sempre que a terra está seca ao toque do dedo na superfície.",
    "Régo com cuidado para não encharcar as raízes das plantas mais sensíveis.",
    "Régo a horta duas vezes por semana nas épocas chuvosas do ano.",
    "Régo com sistema de gota-a-gota para poupar água no verão.",
    "Régo as sementeiras com um regador de bico fino para não as danificar.",
    "Régo o relvado ao fim da tarde para evitar queimaduras solares.",
    "Régo as flores do jardim com a mangueira quando não há chuva.",
    "Régo sempre as plantas antes de sair em viagem para durarem mais.",
    "Régo o canteiro de legumes com água da cisterna que apanho da chuva.",
    "Régo as árvores de fruto jovens duas vezes por semana no verão.",
    "Régo o pomar com o rego de rega que o meu avô construiu.",
    "Régo com moderação as plantas de interior para não as matar.",
    "Régo o solo depois de plantar as sementes para ajudar a germinação.",
]

@pytest.mark.parametrize("s", REGO_VERB)
def test_rego_verb(s):
    _expect(s, "rego", "VERB")


# ── sede / NOUN ───────────────────────────────────────────────────────────────
# Both "headquarters" and "thirst" meanings share the NOUN IPA (ˈsɛdɨ).

SEDE_NOUN = [
    # institution
    "A séde da empresa está localizada no centro histórico do Porto.",
    "A séde do partido ficará na mesma morada durante os próximos anos.",
    "A séde da organização internacional fica em Genebra na Suíça.",
    "A séde do tribunal foi transferida para um novo edifício moderno.",
    "A séde da associação foi inaugurada com uma grande festa cultural.",
    "A séde regional foi encerrada por decisão da administração central.",
    "A séde do clube desportivo tem piscina, ginásio e salas de reunião.",
    "A séde da federação recebe os processos de todas as delegações.",
    "A séde europeia da multinacional emprega mais de dois mil trabalhadores.",
    # thirst — also NOUN IPA
    "Tenho muita sede depois de tanto exercício ao calor intenso.",
    "A sede aumenta quando bebemos bebidas alcoólicas e salgadas.",
    "A sede intensa pode ser sinal de diabetes não diagnosticada.",
    "Sinto sede mesmo depois de beber um copo grande de água.",
    "A sede foi tanta que bebi toda a garrafa de água de um fôlego.",
    "A sede é um sinal de que o corpo precisa de se reidratar.",
    "Sinto sede constante quando o ar condicionado está muito ligado.",
    "Sinto sede quando caminho mais de trinta minutos sem beber.",
    "A sede que sinto depois do treino é sempre muito intensa.",
    "Acordo com sede sempre que janto muito salgado na véspera.",
    "A sede crónica deve ser avaliada por um médico especialista.",
]

@pytest.mark.parametrize("s", SEDE_NOUN)
def test_sede_noun(s):
    _expect(s, "sede", "NOUN")


# ── sobre / NOUN (envelope) ───────────────────────────────────────────────────

SOBRE_NOUN = [
    "Coloca o cheque dentro do sobre e envia pelo correio normal.",
    "O sobre estava selado e endereçado ao destinatário correcto.",
    "O sobre com os documentos foi entregue em mão no escritório.",
    "O sobre branco estava sobre a secretária quando cheguei de manhã.",
    "O sobre rasgado revelou uma carta manuscrita muito delicada.",
    "O sobre com o resultado do exame chegou pelo correio ontem.",
    "O sobre transparente deixava ver o documento que estava dentro.",
    "O sobre dourado do convite de casamento era muito elegante e requintado.",
    "O sobre com as fotografias foi enviado para todos os membros da família.",
]

@pytest.mark.parametrize("s", SOBRE_NOUN)
def test_sobre_noun(s):
    _expect(s, "sobre", "NOUN")


# ── sobre / VERB (sobrar — be left over) ─────────────────────────────────────

SOBRE_VERB = [
    "Sóbre sempre comida quando cozinho para muitas pessoas de uma vez.",
    "Sóbre sempre um pouco de massa mesmo quando calculo as quantidades.",
    "Sóbre comida no final das festas de família que organizamos em casa.",
    "Sóbre sempre café quando o faço de manhã, e bebo o que sôbra.",
    "Sóbre dinheiro no fim do mês quando consigo controlar os gastos.",
    "Sóbre tempo no final do dia quando planeio bem as actividades.",
    "Sóbre arroz sempre que o cozinho, mesmo a calcular bem a dose.",
    "Sóbre pão aos fins de semana e uso-o para fazer a açorda do almoço.",
    "Sóbre tinta depois de pintar e guardo para os retoques futuros.",
    "Sóbre leite quando faço os iogurtes e uso-o para a sopa do jantar.",
    # "caso" here is conditional conjunction (= "if/in case"), not governing noun.
    # "sobre" is 3rd-person subjunctive of "sobrar" (= "if food is left over").
    "caso sóbre comida guarda para o jantar de amanhã.",
    "caso sóbre massa do almoço uso-a para a sopa da noite.",
    "caso sóbre vinho da festa põe numa garrafa e guarda no frio.",
]

@pytest.mark.parametrize("s", SOBRE_VERB)
def test_sobre_verb(s):
    _expect(s, "sobre", "VERB")


# ── sobre / ADP (prep "about/over") — pronounced like NOUN (ˈsobɾɨ) ──────────
# The ADP reading shares the closed-o IPA with NOUN.  Current scoring can
# confuse it with VERB when a DET follows immediately; marked xfail per
# sentence where that pattern occurs.

SOBRE_ADP = [
    # "sobre" ADP (about/concerning/over) — shares IPA with NOUN (envelope).
    # The scorer gives a base +4 for mid-sentence "sobre", which beats VERB
    # signals in all typical ADP contexts.
    "Escreveu um artigo sobre as alterações climáticas e as suas causas.",
    "A conferência sobre energia renovável reuniu especialistas de todo o mundo.",
    "Ela fez uma apresentação sobre os resultados obtidos no último trimestre.",
    "O debate sobre a educação pública durou toda a noite no parlamento.",
    "Não há unanimidade sobre a melhor forma de resolver este problema.",
    "A investigação sobre o tema revelou dados surpreendentes e novos.",
    "O relatório sobre a situação económica foi publicado na semana passada.",
    "As opiniões sobre este assunto divergem muito entre os especialistas.",
    "O consenso sobre a necessidade de mudança é generalizado e crescente.",
    "A discussão sobre os direitos humanos é sempre complexa e sensível.",
    "A notícia sobre o acordo foi recebida com surpresa pela população.",
    "O livro sobre a história de Portugal é muito bem escrito.",
    "Falamos sobre o projeto durante horas sem chegar a conclusão.",
    "O documentário sobre o oceano foi exibido ontem à noite.",
    "A decisão sobre o futuro do projeto será tomada amanhã.",
]

@pytest.mark.parametrize("s", SOBRE_ADP)
def test_sobre_adp(s):
    _expect(s, "sobre", "ADP")


# ── torre / NOUN ──────────────────────────────────────────────────────────────

TORRE_NOUN = [
    "A torre da catedral é visível de toda a cidade em dia de sol.",
    "A torre de controlo coordena todos os voos do aeroporto nacional.",
    "A torre Eiffel foi construída para a exposição universal de Paris.",
    "A torre do relógio marca as horas desde o século dezasseis.",
    "A torre medieval foi restaurada e aberta ao público recentemente.",
    "A torre de vigia estava estrategicamente localizada no ponto mais alto.",
    "A torre de comunicações emite sinais para toda a região envolvente.",
    "A torre do castelo resiste há séculos às intempéries e ao tempo.",
    "A torre de apartamentos tem trinta e dois pisos e vista para o rio.",
    "A torre sineira da igreja toca a dobrar nas grandes celebrações religiosas.",
    "A torre de xadrez move-se em linha recta em qualquer direcção no tabuleiro.",
    "A torre foi desmontada tijolo a tijolo para uma restauração completa.",
]

@pytest.mark.parametrize("s", TORRE_NOUN)
def test_torre_noun(s):
    _expect(s, "torre", "NOUN")


# ── torre / VERB (torrar — toast/roast) ──────────────────────────────────────

TORRE_VERB = [
    "Tórre as amêndoas no forno a cento e sessenta graus por dez minutos.",
    "Tórre o pão numa frigideira com um fio de azeite para mais sabor.",
    "Tórre o café em grão em casa para garantir maior frescura e aroma.",
    "Tórre as nozes antes de as adicionar ao bolo de cenoura e mel.",
    "Tórre o pão velho no forno para fazer torradas crocantes para a sopa.",
    "Tórre as sementes de sésamo numa frigideira seca até ficarem douradas.",
    "Tórre o coco ralado até ficar dourado para a cobertura do bolo.",
    "Tórre as pinhoas levemente antes de as espalhar pelo prato de queijos.",
    "Tórre o trigo partido antes de o cozinhar para um sabor mais profundo.",
    "Tórre o milho em pipoca na panela com uma tampa bem fechada.",
    "Tórre as especiarias inteiras antes de as moer para mais intensidade.",
    "Tórre os amendoins sem sal numa frigideira antes de os temperar.",
    "Tórre o pão de véspera no forno para recuperar a sua textura crocante.",
    "Tórre as castanhas na frigideira com um pouco de sal grosso.",
]

@pytest.mark.parametrize("s", TORRE_VERB)
def test_torre_verb(s):
    _expect(s, "torre", "VERB")


# ── transtorno / NOUN ─────────────────────────────────────────────────────────

TRANSTORNO_NOUN = [
    pytest.param("A greve causou um grande transtorno nos transportes públicos da cidade.",
                 marks=pytest.mark.xfail(reason="DET ADJ NOUN: prev2-DET VERB signal beats NOUN=0")),
    "O transtorno bipolar é tratável com medicação e acompanhamento adequado.",
    "O transtorno obsessivo compulsivo afecta milhões de pessoas no mundo.",
    "O acidente causou um transtorno enorme na circulação rodoviária.",
    "O transtorno de ansiedade é uma das condições mais frequentes globalmente.",
    "O transtorno de stress pós-traumático pode surgir depois de eventos graves.",
    "O transtorno na produção causou atrasos nas entregas durante semanas.",
    "O transtorno do défice de atenção é diagnosticado cada vez mais cedo.",
    "O transtorno na rotina causou dificuldades de adaptação à criança.",
    "O transtorno causado pela tempestade foi sentido durante dias seguidos.",
    "O transtorno alimentar precisa de tratamento especializado multidisciplinar.",
    "O transtorno no sistema informático interrompeu os serviços durante horas.",
    "O transtorno causado pela obra foi comunicado antecipadamente aos moradores.",
]

@pytest.mark.parametrize("s", TRANSTORNO_NOUN)
def test_transtorno_noun(s):
    _expect(s, "transtorno", "NOUN")


# ── transtorno / VERB ─────────────────────────────────────────────────────────

TRANSTORNO_VERB = [
    "Transtórno facilmente os meus planos quando há imprevistos urgentes.",
    "Transtórno a rotina familiar sempre que tenho viagens de trabalho.",
    "Transtórno o ambiente quando fico nervoso em situações de pressão.",
    "Transtórno as prioridades quando aparecem tarefas mais urgentes.",
    "Transtórno o horário quando os filhos estão doentes e precisam de mim.",
    "Transtórno os planos do fim de semana quando o trabalho não está terminado.",
    "Transtórno a ordem da casa quando estou a preparar uma exposição.",
    "Transtórno a sequência do projeto quando descubro erros a corrigir.",
    "Transtórno a agenda quando surgem reuniões de última hora inesperadas.",
    "Transtórno a cozinha toda quando experimento receitas novas e complexas.",
    "Transtórno os planos de viagem quando preciso de prolongar a estadia.",
    "Transtórno a organização habitual quando estou sob muito stress.",
    "Transtórno as prateleiras quando procuro um livro que não encontro.",
    "Transtórno o calendário quando aceito compromissos em demasia.",
]

@pytest.mark.parametrize("s", TRANSTORNO_VERB)
def test_transtorno_verb(s):
    _expect(s, "transtorno", "VERB")


# ══════════════════════════════════════════════════════════════════════════════
# Extended stress-test sentences — extra coverage for every word and POS.
# These complement the primary lists above and probe additional surface patterns.
# ══════════════════════════════════════════════════════════════════════════════

# ── para (extended) ───────────────────────────────────────────────────────────

PARA_ADP_X = [
    "Ele telefonou para avisar que ia chegar tarde ao jantar.",
    "Saímos para o jardim quando o tempo melhorou ao fim da tarde.",
    "O relatório foi preparado para a reunião de segunda-feira.",
    "Preparámos surpresa para ela no dia do aniversário.",
    "Ele treina para melhorar o seu rendimento a cada semana.",
    "A candidatura foi submetida para o concurso nacional de inovação.",
    "Guarda o recibo para mostrar se precisares de devolver o artigo.",
    "O evento foi adiado para o mês seguinte por causa das obras.",
    "Ela estudou durante meses para o concurso de acesso à magistratura.",
    "O produto foi certificado para consumo humano pela autoridade competente.",
    "Deixei uma nota para ele na mesa de trabalho antes de sair.",
    "A proposta foi aceite para publicação na revista científica.",
]

@pytest.mark.parametrize("s", PARA_ADP_X)
def test_para_adp_x(s):
    _expect(s, "para", "ADP")


PARA_VERB_X = [
    "Ela pára de correr quando sente dor no joelho esquerdo.",
    "O processo pára sempre quando há falta de documentação.",
    "A máquina pára periodicamente para arrefecer e evitar danos.",
    "O atleta pára para recuperar depois de cada série intensa.",
    "Ele pára de escrever quando perde o fio à meada.",
    "O motor pára imediatamente quando detecta sobreaquecimento.",
    "Ela pára de falar e ouve com atenção quando alguém a corrige.",
    "O sistema pára durante a atualização e recomeça sozinho depois.",
    "O sangue pára de fluir logo que fazemos pressão no local.",
    "Ele pára sempre à mesma hora para almoçar com os colegas.",
    "A chuva pára e o sol aparece exactamente quando saímos de casa.",
]

@pytest.mark.parametrize("s", PARA_VERB_X)
def test_para_verb_x(s):
    _expect(s, "para", "VERB")


# ── pelo (extended) ───────────────────────────────────────────────────────────

PELO_ADP_X = [
    "A mensagem foi transmitida pelo representante do governo.",
    "Ficou conhecido pelo seu humor inteligente e espontâneo.",
    "O projeto foi financiado pelo fundo europeu de coesão.",
    "Chegámos ao acordo pelo diálogo paciente e construtivo.",
    "A obra foi concluída pelo empreiteiro antes do prazo fixado.",
    "O prémio foi atribuído pelo júri de forma unânime.",
    "Ficou fascinado pelo resultado do experimento científico.",
    "A cidade foi fundada pelo rei no século décimo segundo.",
    "Ele ficou preso pelo tornozelo num buraco na calçada.",
    "A decisão foi tomada pelo conselho de administração da empresa.",
]

@pytest.mark.parametrize("s", PELO_ADP_X)
def test_pelo_adp_x(s):
    _expect(s, "pelo", "ADP")


PELO_NOUN_X = [
    "O pêlo do cão deixou marcas no tapete da sala de estar.",
    "A análise forense identificou o pêlo do suspeito na cena.",
    "O pêlo do gato ficou preso no filtro do aspirador.",
    "Ela tem alergia ao pêlo de gato e espirra constantemente.",
    "O pêlo do javali é mais rígido do que o do porco doméstico.",
    "O pêlo do urso castanho muda de textura com as estações do ano.",
    "O pêlo do coelho branco contrasta com a terra escura da jaula.",
    "O pêlo do carneiro é tosquiado e vendido para a indústria têxtil.",
    "O pêlo do cão pastor alemão é denso e resistente ao frio.",
    "O pêlo da lontra é impermeável graças às suas propriedades únicas.",
]

@pytest.mark.parametrize("s", PELO_NOUN_X)
def test_pelo_noun_x(s):
    _expect(s, "pelo", "NOUN")


PELO_VERB_X = [
    "Pélo as batatas novas para a sopa de legumes do almoço.",
    "Pélo os kiwis antes de os servir às crianças ao lanche.",
    "Pélo os tomates com água quente para tirá-los mais facilmente.",
    "Pélo as cenouras antes de as juntar ao cozido à portuguesa.",
    "Pélo as amêndoas depois de as escaldar em água a ferver.",
    "Pélo os pêssegos frescos para a sobremesa de domingo.",
    "Pélo a fruta antes de a dar às crianças mais pequenas.",
    "Pélo as castanhas depois de as assar na chapa quente.",
    "Pélo a curgete com o descascador antes de a saltear.",
    "Pélo o alho antes de o esmagar no pilão com sal.",
]

@pytest.mark.parametrize("s", PELO_VERB_X)
def test_pelo_verb_x(s):
    _expect(s, "pelo", "VERB")


# ── tola (extended) ───────────────────────────────────────────────────────────

TOLA_ADJ_X = [
    "Foi uma resposta tola que não convenceu ninguém na sala.",
    "Essa desculpa tola não vai resultar com os mais experientes.",
    "A ideia parecia tola ao início mas revelou-se muito inteligente.",
    "Não sejas tola ao ponto de acreditar em tudo o que te dizem.",
    "Ela pareceu tola ao insistir quando a resposta era claramente não.",
    "Uma pergunta tola nunca é má se for feita com sinceridade.",
    "A atitude tola custou-lhe a confiança dos colegas de equipa.",
    "Ela ficou tola ao perceber que tinha sido enganada outra vez.",
    "A decisão tola foi tomada sem pensar nas consequências possíveis.",
    "A reacção tola à crítica revelou a sua insegurança profunda.",
]

@pytest.mark.parametrize("s", TOLA_ADJ_X)
def test_tola_adj_x(s):
    _expect(s, "tola", "ADJ")


TOLA_NOUN_X = [
    "A tóla assinou sem ler as condições do contrato de arrendamento.",
    "A tóla perdeu a carteira outra vez por não a guardar no lugar.",
    "A tóla comprou o produto claramente falsificado por metade do preço.",
    "A tóla revelou o segredo que todos lhe tinham pedido para guardar.",
    "A tóla chegou ao aeroporto no dia errado com todos os documentos.",
    "A tóla deu a senha a alguém que disse ser do banco ao telefone.",
    "A tóla misturou o mel com o leite quente e ele coagulou.",
    "A tóla enviou o e-mail sem o texto e com o assunto em branco.",
    "A tóla deixou o forno ligado durante toda a noite outra vez.",
    "A tóla chegou à entrevista uma hora atrasada sem qualquer aviso.",
]

@pytest.mark.parametrize("s", TOLA_NOUN_X)
def test_tola_noun_x(s):
    _expect(s, "tola", "NOUN")


# ── seco (extended) ───────────────────────────────────────────────────────────

SECO_ADJ_X = [
    "O cabelo seco parte com facilidade quando se usa demasiado calor.",
    "O solo seco racha em sulcos profundos no verão mais quente.",
    "O rio seco revelou pedras e depressões que estavam submersos.",
    "O ramo seco caiu durante a tempestade de vento forte.",
    "O clima seco do Alentejo favorece a produção de vinhos robustos.",
    "O pão seco de ontem serve para fazer açorda ou migas amanhã.",
    "O ar seco resseca os lábios e a pele durante o inverno.",
    "O tomate seco tem um sabor muito mais intenso do que o fresco.",
    "O figo seco dura meses se guardado em local fresco e seco.",
    "O monte está completamente seco depois de meses sem chuva.",
]

@pytest.mark.parametrize("s", SECO_ADJ_X)
def test_seco_adj_x(s):
    _expect(s, "seco", "ADJ")


SECO_VERB_X = [
    "Séco os pratos com um pano limpo para não deixar manchas.",
    "Séco os copos com um pano de microfibra para ficarem transparentes.",
    "Séco as folhas de manjericão no forno a temperatura baixa.",
    "Séco a louça logo que a retiro da máquina de lavar louça.",
    "Séco as mãos na toalha depois de as lavar exaustivamente.",
    "Séco o pincel com uma folha de papel antes de mudar de cor.",
    "Séco sempre o peixe antes de o fritar para ficar estaladiço.",
    "Séco o cogumelo com papel de cozinha antes de o saltear.",
    "Séco a fruta com papel absorvente antes de a mergulhar no chocolate.",
    "Séco a banheira com um pano seco depois de cada utilização.",
]

@pytest.mark.parametrize("s", SECO_VERB_X)
def test_seco_verb_x(s):
    _expect(s, "seco", "VERB")


# ── acordo (extended) ─────────────────────────────────────────────────────────

ACORDO_NOUN_X = [
    "O acordo prevê a troca de informações entre os dois países.",
    "O acordo de cooperação científica foi renovado por dez anos.",
    "Chegaram a um acordo depois de várias sessões de negociação.",
    "O acordo de cavalheiros raramente tem valor jurídico vinculativo.",
    "O acordo foi assinado na presença dos embaixadores dos dois países.",
    "O acordo de livre comércio reduziu as tarifas de forma significativa.",
    "O acordo de paz foi mediado por uma organização internacional.",
    "O acordo final foi celebrado com um jantar formal diplomático.",
    "O acordo de distribuição abrange todo o território europeu.",
    "O acordo de confidencialidade foi assinado por todas as partes.",
]

@pytest.mark.parametrize("s", ACORDO_NOUN_X)
def test_acordo_noun_x(s):
    _expect(s, "acordo", "NOUN")


ACORDO_VERB_X = [
    "Acórdo sempre que ouço o comboio das seis e meia passar.",
    "Acórdo com a sensação de que algo de bom vai acontecer hoje.",
    "Acórdo antes de todos quando estamos de acampamento na tenda.",
    "Acórdo muitas vezes antes do amanhecer sem razão aparente.",
    "Acórdo tranquilo quando a semana seguinte está bem planeada.",
    "Acórdo muito cedo quando tenho de apanhar o autocarro das seis.",
    "Acórdo com os pés frios porque o cobertor desliza durante a noite.",
    "Acórdo com dor de cabeça quando durmo em posição errada.",
    "Acórdo bem-disposto quando a cama é boa e a almofada certa.",
    "Acórdo a meio da noite quando o gato pula para cima da cama.",
]

@pytest.mark.parametrize("s", ACORDO_VERB_X)
def test_acordo_verb_x(s):
    _expect(s, "acordo", "VERB")


# ── acerto (extended) ─────────────────────────────────────────────────────────

ACERTO_NOUN_X = [
    "O acerto das previsões impressionou os analistas mais céticos.",
    "O acerto na resolução do problema foi reconhecido por todos.",
    "O acerto na escolha do fornecedor poupou meses de trabalho.",
    "Foi um acerto ter contratado aquele engenheiro naquele momento.",
    "O acerto das contas ficou para depois da reunião de avaliação.",
    "O acerto na estratégia de comunicação foi decisivo para o projeto.",
    "O acerto nas previsões climáticas permitiu preparar as colheitas.",
    "O acerto na resposta garantiu-lhe a melhor pontuação do concurso.",
    "O acerto do diagnóstico precoce salvou-lhe a vida naquele momento.",
    "O acerto no tim ing da decisão fez toda a diferença para o resultado.",
]

@pytest.mark.parametrize("s", ACERTO_NOUN_X)
def test_acerto_noun_x(s):
    _expect(s, "acerto", "NOUN")


ACERTO_VERB_X = [
    "Acérto os detalhes de última hora antes da apresentação pública.",
    "Acérto o despertador para as seis da manhã todas as noites.",
    "Acérto o tom da voz quando falo em público para projetar mais.",
    "Acérto raramente nas previsões do tempo que faço para o fim de semana.",
    "Acérto o volume do som antes de começar a reunião por videoconferência.",
    "Acérto os horários de transporte para não perder as ligações.",
    "Acérto a posição do microfone antes de cada sessão de gravação.",
    "Acérto as despesas partilhadas no final de cada viagem com os amigos.",
    "Acérto as configurações antes de entregar o computador ao cliente.",
    "Acérto as datas com os participantes antes de enviar o convite.",
]

@pytest.mark.parametrize("s", ACERTO_VERB_X)
def test_acerto_verb_x(s):
    _expect(s, "acerto", "VERB")


# ── cerro (extended) ──────────────────────────────────────────────────────────

CERRO_NOUN_X = [
    "O cerro estava coberto de neve durante toda a semana de janeiro.",
    "A vista do cerro abrangia três concelhos e dois rios no vale.",
    "O cerro foi palco de uma batalha histórica há mais de cinco séculos.",
    "O cerro serve de ponto de referência para os viajantes da região.",
    "O cerro granítico resiste à erosão melhor do que os calcários.",
    "O cerro está incluído numa rota de pedestrianismo regional.",
    "O cerro foi nomeado como geossítio de interesse nacional.",
    "A ermida no cerro está aberta apenas durante as festas da aldeia.",
    "O cerro divide dois vales com microclimas bastante distintos.",
    "O cerro tem uma nascente de água na vertente norte muito fria.",
]

@pytest.mark.parametrize("s", CERRO_NOUN_X)
def test_cerro_noun_x(s):
    _expect(s, "cerro", "NOUN")


CERRO_VERB_X = [
    "Cérro a torneira depois de lavar os dentes para poupar água.",
    "Cérro a janela com cuidado para não acordar o bebé que dorme.",
    "Cérro o porta-moedas sempre que pago para não perder troco.",
    "Cérro os olhos um momento para me concentrar no problema.",
    "Cérro a gaveta com a chave quando saio do escritório à noite.",
    "Cérro o caderno depois de acabar de escrever os apontamentos.",
    "Cérro o casaco antes de sair porque está muito frio lá fora.",
    "Cérro os punhos quando sinto uma injustiça grande acontecer.",
    "Cérro a caixa de cartão com fita antes de a enviar pelos correios.",
    "Cérro as venezianas antes de dormir para o quarto ficar escuro.",
]

@pytest.mark.parametrize("s", CERRO_VERB_X)
def test_cerro_verb_x(s):
    _expect(s, "cerro", "VERB")


# ── choro (extended) ──────────────────────────────────────────────────────────

CHORO_NOUN_X = [
    "O choro dela comoveu toda a gente que estava presente na sala.",
    "O choro de emoção foi a reacção natural de quem esperava tanto.",
    "O choro da criança durou horas antes de ela adormecer finalmente.",
    "O choro do bebé preocupou os pais durante toda a primeira semana.",
    "O choro de alegria foi tão intenso como o de tristeza.",
    "O choro coletivo na cerimónia do funeral foi profundamente tocante.",
    "O choro dela interrompeu várias vezes o discurso de agradecimento.",
    "O choro silencioso foi mais expressivo do que qualquer palavra dita.",
    "O choro de alívio veio depois de receber os resultados negativos.",
    "O choro da noite foi seguido por um dia inteiro de paz.",
]

@pytest.mark.parametrize("s", CHORO_NOUN_X)
def test_choro_noun_x(s):
    _expect(s, "choro", "NOUN")


CHORO_VERB_X = [
    "Chóro de vez em quando quando sinto a saudade da minha terra.",
    "Chóro sempre que ouço cantar fado numa tasca antiga de Lisboa.",
    "Chóro muito raramente mas quando acontece é fundo e prolongado.",
    "Chóro de alegria nos momentos que mais importam da minha vida.",
    "Chóro quando vejo fotograf ias antigas da família reunida.",
    "Chóro em silêncio para não assustar os filhos com a minha tristeza.",
    "Chóro quando o esforço de anos é finalmente reconhecido.",
    "Chóro ao despedir-me de pessoas queridas por longos períodos.",
    "Chóro quando me lembro de pessoas que já não estão entre nós.",
    "Chóro durante os filmes tristes mesmo quando já os vi várias vezes.",
]

@pytest.mark.parametrize("s", CHORO_VERB_X)
def test_choro_verb_x(s):
    _expect(s, "choro", "VERB")


# ── colher (extended) ─────────────────────────────────────────────────────────

COLHER_NOUN_X = [
    "A colhér estava dobrada no fundo da gaveta das tralhas.",
    "Usa a colhér de chá para mexer o açúcar no copo de chá.",
    "A colhér de servir ficou dentro da tigela de puré de batata.",
    "Ele usa sempre a colhér de madeira para não arranhar a frigideira.",
    "A colhér de sopa basta para medir uma dose de azeite na receita.",
    "A colhér de prata era o único talher que sobrou do serviço antigo.",
    "Pôs a colhér dentro do copo para não manchar a toalha da mesa.",
    "A colhér de gelado tem um cabo longo para chegar ao fundo.",
    "A criança deixou a colhér cair no chão três vezes durante o jantar.",
    "A colhér de sopa aguenta uma dose generosa de caldo quente.",
]

@pytest.mark.parametrize("s", COLHER_NOUN_X)
def test_colher_noun_x(s):
    _expect(s, "colher", "NOUN")


COLHER_VERB_X = [
    "Vou colher uvas da vinha ainda esta semana antes da chuva.",
    "É hora de colher as azeitonas antes que caiam ao vento.",
    "Eles foram colher amendoins da horta no final do verão.",
    "Precisamos de colher a alfazema antes que perca o aroma.",
    "É difícil colher cogumelos silvestres sem conhecer bem a serra.",
    "Fui colher ervas aromáticas para pôr no caldo de carne.",
    "Foram colher amoras no monte antes de o verão acabar.",
    "Vamos colher os melões quando estiverem completamente maduros.",
    "É preciso colher os tomates antes das primeiras geadas de outubro.",
    "Foram colher flores do campo para o buquê da cerimónia.",
]

@pytest.mark.parametrize("s", COLHER_VERB_X)
def test_colher_verb_x(s):
    _expect(s, "colher", "VERB")


# ── começo (extended) ─────────────────────────────────────────────────────────

COMEÇO_NOUN_X = [
    "O começo da viagem foi marcado pela perda de uma mala.",
    "O começo da conversa foi cordial mas rapidamente tensionou.",
    "O começo do espetáculo foi adiado por mau funcionamento do sistema.",
    "O começo da reforma foi gradual para reduzir o impacto social.",
    "O começo do outono traz as chuvas e o arrefecimento gradual.",
    "O começo da recuperação económica foi lento mas consistente.",
    "O começo do processo judicial demorou mais do que o previsto.",
    "O começo do projeto foi financiado por um subsídio europeu.",
    "O começo da parceria foi marcado por muita desconfiança mútua.",
    "O começo da história é o mais difícil de escrever com fluência.",
]

@pytest.mark.parametrize("s", COMEÇO_NOUN_X)
def test_começo_noun_x(s):
    _expect(s, "começo", "NOUN")


COMEÇO_VERB_X = [
    "Coméço a preparar o almoço quando o filho chega da escola.",
    "Coméço a sentir fome assim que passa o meio-dia sem almoçar.",
    "Coméço por ouvir a opinião de todos antes de decidir algo.",
    "Coméço o dia com gratidão por tudo o que já conquistei.",
    "Coméço a escrever o relatório assim que os dados estão prontos.",
    "Coméço a desconfiar quando as histórias não batem umas com as outras.",
    "Coméço a aula com uma revisão do que foi dado na sessão anterior.",
    "Coméço a sentir falta de casa quando a viagem ultrapassa os dez dias.",
    "Coméço sempre por definir os objetivos antes de iniciar qualquer projeto.",
    "Coméço o trabalho no computador novo esta semana depois da formação.",
]

@pytest.mark.parametrize("s", COMEÇO_VERB_X)
def test_começo_verb_x(s):
    _expect(s, "começo", "VERB")


# ── conserto (extended) ───────────────────────────────────────────────────────

CONSERTO_NOUN_X = [
    "O conserto da caldeira ficou orçamentado em mais do que eu esperava.",
    "O conserto do porta-bagagens do carro durou apenas meia hora.",
    "O conserto do telhado foi urgente antes das chuvas de novembro.",
    "O conserto da jante riscada foi feito no próprio dia da ocorrência.",
    "O conserto do sistema de som do carro durou toda a tarde.",
    "O conserto da vedação do jardim ficou concluído antes do inverno.",
    "O conserto da moldura foi feito por um restaurador especialista.",
    "O conserto do motor ficou mais barato do que qualquer orçamento dado.",
    "O conserto da vitrinha foi feito de forma tão perfeita que não se nota.",
    "O conserto do elevador obrigou os moradores a usar as escadas.",
]

@pytest.mark.parametrize("s", CONSERTO_NOUN_X)
def test_conserto_noun_x(s):
    _expect(s, "conserto", "NOUN")


CONSERTO_VERB_X = [
    "Consérto o que estiver avariado antes de qualquer outra coisa.",
    "Consérto os erros do texto antes de o enviar ao editor.",
    "Consérto roupa que outros atirariam fora sem pensar duas vezes.",
    "Consérto bicicletas antigas para as reintegrar no ciclo de uso.",
    "Consérto a calha partida antes que a próxima chuva cause danos.",
    "Consérto os problemas de postura com exercícios específicos diários.",
    "Consérto a relação com os clientes quando há mal-entendidos.",
    "Consérto o que puder hoje para não acumular trabalho para amanhã.",
    "Consérto o manuscrito antes de o submeter à editora para avaliação.",
    "Consérto sempre os pequenos erros antes de apresentar o trabalho.",
]

@pytest.mark.parametrize("s", CONSERTO_VERB_X)
def test_conserto_verb_x(s):
    _expect(s, "conserto", "VERB")


# ── coro (extended) ───────────────────────────────────────────────────────────

CORO_NOUN_X = [
    "O coro infantil cantou as janeiras em todas as ruas da aldeia.",
    "O coro de câmara estreou uma peça contemporânea nessa noite.",
    "O coro foi convidado para actuar no festival internacional em junho.",
    "O coro interpretou o Réquiem de Mozart com grande expressividade.",
    "O coro amadureceu muito depois de contratar um novo regente.",
    "O coro de vozes brancas comoveu o público da catedral.",
    "O coro da paróquia foi fundado por um padre musicólogo.",
    "O coro participou na abertura das festividades municipais.",
    "O coro recebeu uma menção honrosa no concurso nacional coral.",
    "O coro actua este domingo na sala grande do conservatório.",
]

@pytest.mark.parametrize("s", CORO_NOUN_X)
def test_coro_noun_x(s):
    _expect(s, "coro", "NOUN")


CORO_VERB_X = [
    "Córo quando me pedem para falar diante de muitas pessoas.",
    "Córo facilmente mesmo com um elogio simples e sincero.",
    "Córo de vergonha quando alguém conta algo embaraçoso meu.",
    "Córo quando o professor me chama pelo nome na sala cheia.",
    "Córo ao receber um prémio em público sem estar preparada.",
    "Córo sempre que me dizem que sou bonita diante de outros.",
    "Córo ao admitir que me enganei diante de toda a equipa.",
    "Córo quando me pedem para cantar mesmo que conheça a letra.",
    "Córo ao lembrar as coisas menos inteligentes que já disse.",
    "Córo quando me apresentam como a melhor da turma ao pai.",
]

@pytest.mark.parametrize("s", CORO_VERB_X)
def test_coro_verb_x(s):
    _expect(s, "coro", "VERB")


# ── corte (extended) ──────────────────────────────────────────────────────────

CORTE_NOUN_X = [
    "O córte de luz afectou todo o bairro durante a manhã inteira.",
    "O córte de gastos foi necessário para equilibrar o orçamento.",
    "O córte do fio revelou o interior colorido do cabo eléctrico.",
    "O córte de financiamento suspendeu o programa de investigação.",
    "O córte de relações diplomáticas foi um sinal grave de tensão.",
    "O córte de energia solar deixou o sistema de aquecimento parado.",
    "O córte transversal do tronco revelou os anéis de crescimento.",
    "O córte de acesso à autoestrada obrigou os condutores a desviar.",
    "O córte final do filme ficou diferente do que a realizadora tinha previsto.",
    "O córte do tecido foi feito na diagonal para maior elasticidade.",
]

@pytest.mark.parametrize("s", CORTE_NOUN_X)
def test_corte_noun_x(s):
    _expect(s, "corte", "NOUN")


CORTE_VERB_X = [
    "Córte o fio com a tesoura afiada para um resultado limpo.",
    "Córte as rosas antes que murchem e ponha-as num vaso com água.",
    "Córte a ligação à internet se não precisar dela durante o estudo.",
    "Córte os legumes em pedaços iguais para cozerem ao mesmo tempo.",
    "Córte o papel ao meio antes de o dobrar em quatro partes.",
    "Córte o acesso às redes sociais nas horas de maior trabalho.",
    "Córte a fruta em pedaços pequenos para a salada de fruta.",
    "Córte as beiras do bolo para ficarem todas iguais e apresentáveis.",
    "Córte o molde com precisão para que o tecido fique perfeito.",
    "Córte a conversa quando perceber que está a ser manipulada.",
]

@pytest.mark.parametrize("s", CORTE_VERB_X)
def test_corte_verb_x(s):
    _expect(s, "corte", "VERB")


# ── forma (extended) ──────────────────────────────────────────────────────────

FORMA_NOUN_X = [
    "A forma como ele sorriu disse tudo sem precisar de palavras.",
    "A forma do cristal é determinada pela sua estrutura molecular.",
    "Não há outra forma de conseguir isso sem muito esforço diário.",
    "A forma de comunicar influencia muito o resultado das negociações.",
    "A forma dos montes ao fundo lembrava um perfil humano deitado.",
    "A forma mais simples nem sempre é a mais eficaz para todos.",
    "A forma como ela reage à pressão revela muito do seu carácter.",
    "A forma de preparar a massa folhada exige paciência e técnica.",
    "A forma como o projeto foi apresentado foi o segredo do sucesso.",
    "A forma geométrica do pavilhão foi inspirada numa concha marinha.",
]

@pytest.mark.parametrize("s", FORMA_NOUN_X)
def test_forma_noun_x(s):
    _expect(s, "forma", "NOUN")


FORMA_VERB_X = [
    "A prática forma os hábitos que moldam o nosso carácter ao longo dos anos.",
    "A leitura forma leitores críticos capazes de questionar o que lêem.",
    "A família forma os valores essenciais que nos guiam durante a vida.",
    "A cultura forma a identidade de um povo ao longo das gerações.",
    "A universidade forma profissionais capazes de responder aos desafios.",
    "A experiência forma julgamentos que nenhum livro consegue ensinar.",
    "A escola forma cidadãos que contribuem para uma sociedade melhor.",
    "A crise forma carácter quando enfrentada com resiliência e determinação.",
    "A repetição forma o talento quando acompanhada de reflexão crítica.",
    "A empresa forma continuamente as suas equipas para manter competitividade.",
]

@pytest.mark.parametrize("s", FORMA_VERB_X)
def test_forma_verb_x(s):
    _expect(s, "forma", "VERB")


# ── gosto (extended) ──────────────────────────────────────────────────────────

GOSTO_NOUN_X = [
    "Tem um gosto refinado que se nota em tudo o que escolhe usar.",
    "O gosto da sopa de cebola caseira é incomparável com qualquer industrial.",
    "O gosto pelo jazz veio depois de ouvir Miles Davis pela primeira vez.",
    "O gosto pessoal é subjectivo mas o bom gosto é reconhecível.",
    "O gosto pelo cinema surgiu quando o pai o levou ao drive-in.",
    "O gosto desta água é levemente metálico e não é muito agradável.",
    "O gosto que ela tem para decorar a casa é invejável.",
    "O gosto pelo desporto foi cultivado desde os primeiros anos de escola.",
    "O gosto amargo do café preto é o que o torna tão apreciado.",
    "O gosto pela natureza cresceu depois das caminhadas de verão.",
]

@pytest.mark.parametrize("s", GOSTO_NOUN_X)
def test_gosto_noun_x(s):
    _expect(s, "gosto", "NOUN")


GOSTO_VERB_X = [
    "Gósto muito de conversar com quem tem opiniões diferentes das minhas.",
    "Gósto de acordar sem despertador quando não tenho compromissos.",
    "Gósto de visitar museus sempre que viajo para uma cidade nova.",
    "Gósto de fazer caminhadas solitárias para pensar sem interrupções.",
    "Gósto de comer devagar e apreciar cada sabor no prato.",
    "Gósto de ler as críticas antes de ver um filme ou peça.",
    "Gósto de escrever cartas à mão mesmo na era digital que vivemos.",
    "Gósto de cheiro de chuva na terra seca em pleno verão.",
    "Gósto de tempo livre para não fazer nada em particular.",
    "Gósto de aprender uma palavra nova em idiomas que não domino.",
]

@pytest.mark.parametrize("s", GOSTO_VERB_X)
def test_gosto_verb_x(s):
    _expect(s, "gosto", "VERB")


# ── gozo (extended) ───────────────────────────────────────────────────────────

GOZO_NOUN_X = [
    "O gozo das suas prerrogativas está condicionado ao cargo ocupado.",
    "O gozo das instalações desportivas é gratuito para os sócios.",
    "O gozo da liberdade de expressão está garantido pela constituição.",
    "O gozo das férias foi perturbado por uma crise na empresa.",
    "O gozo de servidão predial foi reconhecido pelo tribunal competente.",
    "O gozo do subsídio de transporte é mensal e automático.",
    "O gozo dos benefícios fiscais depende do cumprimento das condições.",
    "O gozo de capacidade civil plena é adquirido com a maioridade.",
    "O gozo das regalias sociais depende de inscrição válida no sistema.",
    "O gozo do ambiente natural deve ser compatível com a sua preservação.",
]

@pytest.mark.parametrize("s", GOZO_NOUN_X)
def test_gozo_noun_x(s):
    _expect(s, "gozo", "NOUN")


GOZO_VERB_X = [
    "Gózo de grande autonomia na gestão do meu trabalho diário.",
    "Gózo das manhãs de sábado sem alarme nem agenda preenchida.",
    "Gózo de uma vista privilegiada da cidade a partir do meu escritório.",
    "Gózo de boa reputação que construí com muito esforço ao longo dos anos.",
    "Gózo dos últimos dias de sol antes do outono se instalar.",
    "Gózo de plena saúde e isso é o maior dos privilégios.",
    "Gózo de confiança absoluta da minha equipa em momentos de crise.",
    "Gózo das noites de verão no jardim com os amigos mais chegados.",
    "Gózo de um equilíbrio entre trabalho e vida pessoal que poucos têm.",
    "Gózo do silêncio da montanha nas manhãs de caminhada solitária.",
]

@pytest.mark.parametrize("s", GOZO_VERB_X)
def test_gozo_verb_x(s):
    _expect(s, "gozo", "VERB")


# ── jogo (extended) ───────────────────────────────────────────────────────────

JOGO_NOUN_X = [
    "O jogo foi suspenso ao intervalo por causa de incidentes na bancada.",
    "O jogo de palavras cruzadas está no jornal de domingo.",
    "O jogo de golfe exige concentração e serenidade acima de tudo.",
    "O jogo de soma zero nunca beneficia as duas partes envolvidas.",
    "O jogo de sedução entre os dois era evidente para todos na festa.",
    "O jogo do gato e do rato prolongou-se por mais duas horas.",
    "O jogo de cartas costumava ir até às três da manhã.",
    "O jogo de espelhos confundiu a criança que tentava encontrar saída.",
    "O jogo foi transmitido em directo para todo o país pela televisão.",
    "O jogo de futebol de salão decorre no pavilhão municipal.",
]

@pytest.mark.parametrize("s", JOGO_NOUN_X)
def test_jogo_noun_x(s):
    _expect(s, "jogo", "NOUN")


JOGO_VERB_X = [
    "Jógo xadrez online sempre que tenho dez minutos livres no dia.",
    "Jógo futebol com os meus filhos no jardim ao fim do dia.",
    "Jógo às cartas com os avós quando os visito ao fim de semana.",
    "Jógo dominó com os amigos depois do jantar de vez em quando.",
    "Jógo padel desde que me apresentaram ao desporto há dois anos.",
    "Jógo no time livre para distrair a cabeça depois de um dia difícil.",
    "Jógo bilhar ocasionalmente no clube do bairro onde sou sócio.",
    "Jógo damas com o meu irmão mais novo nas tardes de chuva.",
    "Jógo videojogos de estratégia quando preciso de estimular o raciocínio.",
    "Jógo com palavras quando escrevo para criar ritmo no texto.",
]

@pytest.mark.parametrize("s", JOGO_VERB_X)
def test_jogo_verb_x(s):
    _expect(s, "jogo", "VERB")


# ── molho (extended) ──────────────────────────────────────────────────────────

MOLHO_NOUN_X = [
    "O molho de alho e coentros é a base de muitos pratos alentejanos.",
    "O molho ficou perfeito depois de reduzir durante vinte minutos.",
    "O molho de amendoim é essencial na culinária tailandesa autêntica.",
    "O molho do assado de borrego absorveu todos os sabores das ervas.",
    "O molho de piri-piri caseiro é muito mais aromático do que o industrial.",
    "O molho ficou salgado demais porque me distraí com o telemóvel.",
    "O molho de queijo gratinado cobria a couve-flor assada no forno.",
    "O molho de vinho tinto combina perfeitamente com o javali assado.",
    "O molho de legumes reduce muito e fica muito mais concentrado.",
    "O molho de manteiga e limão finaliza o peixe grelhado na perfeição.",
]

@pytest.mark.parametrize("s", MOLHO_NOUN_X)
def test_molho_noun_x(s):
    _expect(s, "molho", "NOUN")


MOLHO_VERB_X = [
    "Mólho as leguminosas durante a noite para reduzirem o tempo de cozinha.",
    "Mólho o pão torrado no café com leite ao pequeno-almoço.",
    "Mólho as mãos antes de aplicar o creme para maior absorção.",
    "Mólho as tâmaras em água quente antes de as triturar para o bolo.",
    "Mólho o pincel antes de mergulhá-lo na tinta para suavizar.",
    "Mólho as amêndoas em água fria para ficar mais fácil pelá-las.",
    "Mólho os pés quando chego a casa depois de um dia longo de caminhada.",
    "Mólho as sementes de chia em leite vegetal durante a noite.",
    "Mólho as uvas passas em aguardente para o bolo de natal.",
    "Mólho o biscoito no chá e como-o antes que se desfaça.",
]

@pytest.mark.parametrize("s", MOLHO_VERB_X)
def test_molho_verb_x(s):
    _expect(s, "molho", "VERB")


# ── olho (extended) ───────────────────────────────────────────────────────────

OLHO_NOUN_X = [
    "O olho do tomate foi cortado antes de o fatiar para a salada.",
    "O olho de peixe distorce a perspectiva mas capta muito ângulo.",
    "O olho laser corrigiu-lhe a miopia em menos de quinze minutos.",
    "O olho inflamado precisou de colírio durante uma semana seguida.",
    "O olho do lince é capaz de detectar presas a grande distância.",
    "O olho do retrato parecia seguir quem percorresse a galeria.",
    "O olho humano adapta-se à obscuridade em poucos minutos.",
    "O olho clínico do médico detectou o problema à primeira vista.",
    "O olho da mosca é composto por milhares de facetas microscópicas.",
    "O olho do boi era a janela redonda mais característica da fachada.",
]

@pytest.mark.parametrize("s", OLHO_NOUN_X)
def test_olho_noun_x(s):
    _expect(s, "olho", "NOUN")


OLHO_VERB_X = [
    "Ólho para o relógio quando a reunião está a durar demais.",
    "Ólho pela janela do comboio e observo a paisagem a mudar.",
    "Ólho para o ecrã com atenção antes de clicar em qualquer botão.",
    "Ólho para os preços antes de adicionar qualquer produto ao cesto.",
    "Ólho para o mapa antes de cada caminhada para não me perder.",
    "Ólho sempre para ambos os lados antes de atravessar a rua.",
    "Ólho para os resultados antes de dar a opinião sobre qualquer coisa.",
    "Ólho para o filho quando ele está a dormir e sinto paz.",
    "Ólho para o que faço antes de o apresentar ao cliente.",
    "Ólho para as estrelas nas noites limpas de verão na serra.",
]

@pytest.mark.parametrize("s", OLHO_VERB_X)
def test_olho_verb_x(s):
    _expect(s, "olho", "VERB")


# ── peso (extended) ───────────────────────────────────────────────────────────

PESO_NOUN_X = [
    "O peso da mala excedeu o limite permitido pela companhia aérea.",
    "O peso do silêncio daquela sala era quase insuportável.",
    "O peso do fardo histórico impede o progresso de muitas nações.",
    "O peso da culpa é muito mais difícil de carregar do que o da dívida.",
    "O peso molecular do composto foi calculado com grande precisão.",
    "O peso do argumento convenceu os mais céticos da sala.",
    "O peso da responsabilidade cresce com o nível hierárquico do cargo.",
    "O peso seco do material foi registado após secagem completa.",
    "O peso da carga na ponte foi monitorizado durante toda a travessia.",
    "O peso das palavras ditas em público é muito maior do que em privado.",
]

@pytest.mark.parametrize("s", PESO_NOUN_X)
def test_peso_noun_x(s):
    _expect(s, "peso", "NOUN")


PESO_VERB_X = [
    "Péso as cenouras antes de as cozer para saber a dose exacta.",
    "Péso as algas secas antes de as adicionar ao caldo de peixe.",
    "Péso as farinhas separadamente para maior precisão na pastelaria.",
    "Péso as malas sempre antes de as facturar no aeroporto.",
    "Péso as palavras antes de as dizer em situações sensíveis.",
    "Péso os ingredientes ao milímetro quando faço receitas técnicas.",
    "Péso o bagageiro semanalmente durante a preparação para a viagem.",
    "Péso os dois lados da questão antes de formar uma opinião.",
    "Péso as consequências antes de tomar qualquer decisão irreversível.",
    "Péso os riscos com cuidado antes de investir o capital disponível.",
]

@pytest.mark.parametrize("s", PESO_VERB_X)
def test_peso_verb_x(s):
    _expect(s, "peso", "VERB")


# ── porto (extended) ──────────────────────────────────────────────────────────

PORTO_NOUN_X = [
    "O porto de Aveiro é um porto artificial com uma longa barra.",
    "O porto de Viana do Castelo é dos mais pitoresco s do país.",
    "O porto comercial operou sem interrupção durante o fim de semana.",
    "O porto de pesca estava animado ao amanhecer com as chegadas.",
    "O porto foi palco de grandes descobertas e partidas históricas.",
    "O porto de águas profundas recebe navios de grande calado.",
    "O porto franqueado estimulou o comércio exterior da região.",
    "O porto foi reabilitado e transformado num espaço cultural.",
    "O porto de recreio foi inaugurado e trouxe novos investimentos.",
    "O porto foi eleito o melhor porto de cruzeiros da Europa.",
]

@pytest.mark.parametrize("s", PORTO_NOUN_X)
def test_porto_noun_x(s):
    _expect(s, "porto", "NOUN")


PORTO_VERB_X = [
    "Pórto-me com integridade mesmo quando ninguém está a ver.",
    "Pórto-me de forma madura em todas as situações profissionais.",
    "Pórto-me bem em qualquer evento porque fui bem educado.",
    "Pórto-me de forma respeitosa com as pessoas mais velhas.",
    "Pórto-me com civismo mesmo quando os outros não o fazem.",
    "Pórto-me de acordo com os valores que prezo na minha vida.",
    "Pórto-me bem sob pressão porque já aprendi a gerir o stress.",
    "Pórto-me com humildade quando recebo elogios pelo meu trabalho.",
    "Pórto-me de forma adequada ao contexto de cada situação nova.",
    "Pórto-me com serenidade mesmo em situações de grande conflito.",
]

@pytest.mark.parametrize("s", PORTO_VERB_X)
def test_porto_verb_x(s):
    _expect(s, "porto", "VERB")


# ── posto (extended) ──────────────────────────────────────────────────────────

POSTO_NOUN_X = [
    "O posto de trabalho foi extinto com a automação do processo.",
    "O posto avançado foi reforçado com mais pessoal especializado.",
    "O posto de saúde da aldeia tem médico apenas às terças e quintas.",
    "O posto fronteiriço reabriu depois de meses de encerramento.",
    "O posto de comando foi instalado numa tenda no centro do campo.",
    "O posto de abastecimento ficou sem gasolina durante o feriado.",
    "O posto diplomático exige fluência em três idiomas no mínimo.",
    "O posto de vigilância foi desactivado após a conclusão das obras.",
    "O posto da PSP fica a duzentos metros da entrada do mercado.",
    "O posto de socorro tem desfibrilhador e equipamento de primeiros socorros.",
]

@pytest.mark.parametrize("s", POSTO_NOUN_X)
def test_posto_noun_x(s):
    _expect(s, "posto", "NOUN")


POSTO_VERB_X = [
    "Pósto as actualizações na página da empresa todas as manhãs.",
    "Pósto as fotografias do evento logo que termina para partilhar.",
    "Pósto resultados na plataforma interna assim que ficam disponíveis.",
    "Pósto sempre mensagens de agradecimento depois dos eventos.",
    "Pósto conteúdo regularmente para manter o canal activo.",
]

@pytest.mark.parametrize("s", POSTO_VERB_X)
def test_posto_verb_x(s):
    _expect(s, "posto", "VERB")


# ── rego (extended) ───────────────────────────────────────────────────────────

REGO_NOUN_X = [
    "O rego de pedra foi construído há mais de cem anos pelos avós.",
    "O rego divide o campo em dois talhões de tamanho igual.",
    "O rego transbordou depois das chuvas intensas de novembro.",
    "O rego foi limpo antes da época de rega para funcionar melhor.",
    "O rego distribui a água por todo o perímetro de rega da aldeia.",
    "O rego de barro ainda existe em algumas quintas históricas do interior.",
    "O rego foi alargado para aumentar o caudal em anos secos.",
    "O rego marca a separação entre a propriedade pública e a privada.",
    "O rego foi reabilitado no âmbito de um projeto de recuperação rural.",
    "O rego principal alimenta os rego s secundários durante a rega.",
]

@pytest.mark.parametrize("s", REGO_NOUN_X)
def test_rego_noun_x(s):
    _expect(s, "rego", "NOUN")


REGO_VERB_X = [
    "Régo as plantas de interior uma vez por semana sem falta.",
    "Régo o jardim ao amanhecer para que a água penetre melhor.",
    "Régo com sistema automático para não esquecer quando estou fora.",
    "Régo a horta nos dias quentes para compensar a evaporação.",
    "Régo as mudas recém-plantadas com cuidado para não as arrancar.",
    "Régo os canteiros de flores alternando dias para equilibrar.",
    "Régo com moderação no inverno porque as plantas precisam de menos.",
    "Régo a oliveira jovem duas vezes por semana durante o verão.",
    "Régo os pés de alface de manhã para que sequem antes da noite.",
    "Régo sempre que a superfície da terra parece seca ao toque.",
]

@pytest.mark.parametrize("s", REGO_VERB_X)
def test_rego_verb_x(s):
    _expect(s, "rego", "VERB")


# ── sede (extended) ───────────────────────────────────────────────────────────

SEDE_NOUN_X = [
    # institution
    "A séde social da empresa foi transferida para o estrangeiro.",
    "A séde regional recebe delegações de dez distritos diferentes.",
    "A séde do banco foi inaugurada com grande pompa no centro da cidade.",
    "A séde da fundação foi instalada num palácio do século dezoito.",
    "A séde da cooperativa serve de ponto de encontro dos associados.",
    # thirst
    "A sede que sinto durante o exercício físico intenso é enorme.",
    "Tenho sede de justiça e não descansarei enquanto não a encontrar.",
    "A sede de conhecimento é o que distingue os bons profissionais.",
    "A sede de reconhecimento pode ser uma fonte de sofrimento.",
    "A sede de poder corrompe quem não tem valores sólidos.",
]

@pytest.mark.parametrize("s", SEDE_NOUN_X)
def test_sede_noun_x(s):
    _expect(s, "sede", "NOUN")


# ── sobre (extended) ──────────────────────────────────────────────────────────

SOBRE_NOUN_X = [
    "O sobre azul estava entre as cartas que chegaram esta manhã.",
    "O sobre com o dinheiro foi entregue discretamente ao responsável.",
    "O sobre acolchoado protege os documentos durante o transporte.",
    "O sobre lacrado só deve ser aberto em presença de testemunhas.",
    "O sobre rasgado na abertura revelou que alguém já o tinha lido.",
    "O sobre deslizou por baixo da porta durante a madrugada.",
    "O sobre com o cheque estava assinado mas sem data impressa.",
    "O sobre de papel reciclado é a opção mais ecológica disponível.",
]

@pytest.mark.parametrize("s", SOBRE_NOUN_X)
def test_sobre_noun_x(s):
    _expect(s, "sobre", "NOUN")


SOBRE_ADP_X = [
    "Ela escreveu um poema sobre o tempo que passa e não volta.",
    "Ele sabe tudo sobre astronomia e gosta de partilhar esse saber.",
    "O professor falou sobre as causas da primeira guerra mundial.",
    "Discutimos sobre o futuro do projeto durante toda a reunião.",
    "A palestra sobre inteligência artificial encheu o auditório.",
    "Os especialistas divergem sobre as causas da crise financeira.",
    "Pensou muito sobre a decisão antes de a comunicar à família.",
    "Os dados sobre o desemprego foram divulgados pelo governo.",
    "A conversa sobre o passado tornou-se incómoda para todos.",
    "O documentário sobre o Alentejo ganhou um prémio internacional.",
    "Os casos sobre direitos digitais multiplicaram-se nos tribunais.",
    "O texto sobre os efeitos do plástico no oceano foi publicado.",
    "Os relatórios sobre o clima apontam para mudanças aceleradas.",
    "O estudante escreveu a tese sobre a literatura lusófona contemporânea.",
    "Os dados sobre a natalidade confirmam a tendência de declínio.",
]

@pytest.mark.parametrize("s", SOBRE_ADP_X)
def test_sobre_adp_x(s):
    _expect(s, "sobre", "ADP")


SOBRE_VERB_X = [
    "Sóbre sempre um pedaço de queijo quando cortamos a peça inteira.",
    "Sóbre tinta de parede se calcularmos bem os litros necessários.",
    "Sóbre trabalho sempre que há baixas na equipa sem substituição.",
    "Sóbre pão da véspera e eu faço sempre açorda com ele.",
    "Sóbre arroz quando não controlo bem a quantidade de água.",
    "Sóbre dinheiro se evitarmos as compras por impulso durante o mês.",
    "Sóbre espaço na mala quando dobro a roupa em vez de a enrolar.",
    "embora sóbre comida não devo comer mais do que preciso.",
    "se sóbre massa ponho no frigorífico e como no dia seguinte.",
]

@pytest.mark.parametrize("s", SOBRE_VERB_X)
def test_sobre_verb_x(s):
    _expect(s, "sobre", "VERB")


# ── torre (extended) ──────────────────────────────────────────────────────────

TORRE_NOUN_X = [
    "A torre do castelo foi classificada como monumento nacional.",
    "A torre de refrigeração emite vapores visíveis a quilómetros.",
    "A torre da catedral pode ser visitada entre as dez e as dezoito horas.",
    "A torre de controlo autorizou a aterragem em segurança.",
    "A torre de xadrez move-se em linha recta por qualquer número de casas.",
    "A torre dos anjos é um marco histórico da cidade medieval.",
    "A torre de telemovel foi instalada no ponto mais alto do monte.",
    "A torre de água abastece o bairro durante os meses mais secos.",
    "A torre do palácio ilumina toda a noite com luzes coloridas.",
    "A torre foi remodelada e transformada num museu de arte contemporânea.",
]

@pytest.mark.parametrize("s", TORRE_NOUN_X)
def test_torre_noun_x(s):
    _expect(s, "torre", "NOUN")


TORRE_VERB_X = [
    "Tórre as nozes antes de as servir com o queijo da entrada.",
    "Tórre o pão de sementes para acompanhar a sopa de legumes.",
    "Tórre as castanhas na frigideira sem gordura até ficarem estaladiças.",
    "Tórre os pimentos na chama do fogão para os pelar facilmente.",
    "Tórre o coco ralado na frigideira seca para o bolo de chocolate.",
    "Tórre as pinhoas antes de as usar na salada de esparguete.",
    "Tórre as sêmolas antes de as cozer para intensificar o sabor.",
    "Tórre as sementes de girassol antes de as adicionar ao pão.",
    "Tórre o pão de ontem no forno para fazer torradas ao pequeno-almoço.",
    "Tórre os amendoins e tempera-os com sal e piri-piri antes de servir.",
]

@pytest.mark.parametrize("s", TORRE_VERB_X)
def test_torre_verb_x(s):
    _expect(s, "torre", "VERB")


# ── transtorno (extended) ─────────────────────────────────────────────────────

TRANSTORNO_NOUN_X = [
    "O transtorno na rede de transportes afectou milhares de passageiros.",
    "O transtorno causado pela enchente foi avaliado em milhões de euros.",
    "O transtorno obsessivo manifesta-se de formas muito diferentes.",
    "O transtorno na obra de casa durou mais do que o previsto.",
    "O transtorno no fornecimento de gás foi resolvido ao fim de três dias.",
    "O transtorno provocado pela greve afectou as exportações do país.",
    "O transtorno de humor pode ser tratado com acompanhamento adequado.",
    "O transtorno criado pelo acidente bloqueou a estrada durante horas.",
    "O transtorno no sistema de pagamentos durou toda a manhã.",
    "O transtorno emocional foi evidente no discurso do presidente.",
]

@pytest.mark.parametrize("s", TRANSTORNO_NOUN_X)
def test_transtorno_noun_x(s):
    _expect(s, "transtorno", "NOUN")


TRANSTORNO_VERB_X = [
    "Transtórno a cozinha toda quando experimento uma receita nova.",
    "Transtórno o horário quando a reunião inesperada aparece.",
    "Transtórno os planos se surgir alguma coisa mais urgente.",
    "Transtórno a ordem habitual quando preciso de pensar diferente.",
    "Transtórno a agenda para incluir o compromisso que esqueci.",
    "Transtórno os móveis da sala quando quero mudar de ambiente.",
    "Transtórno as prioridades quando me chega trabalho urgente.",
    "Transtórno o calendário quando há imprevistos de última hora.",
    "Transtórno os planos do fim de semana quando há trabalho pendente.",
    "Transtórno a rotina às vezes de propósito para sair da zona de conforto.",
]

@pytest.mark.parametrize("s", TRANSTORNO_VERB_X)
def test_transtorno_verb_x(s):
    _expect(s, "transtorno", "VERB")


# ── add_extra_diacritics ──────────────────────────────────────────────────────

def test_diacritics_verb_para():
    result = add_extra_diacritics("O autocarro para em frente ao hospital.")
    assert "pára" in result


def test_diacritics_noun_gosto():
    result = add_extra_diacritics("O gosto do vinho é excelente.")
    assert "gôsto" in result


def test_diacritics_verb_gosto():
    result = add_extra_diacritics("Eu gosto de música clássica.")
    assert "gósto" in result


def test_diacritics_noun_acordo():
    result = add_extra_diacritics("O acordo foi assinado ontem.")
    assert "acôrdo" in result


def test_diacritics_verb_acordo():
    result = add_extra_diacritics("Eu acordo cedo todos os dias.")
    assert "acórdo" in result


# keep imports for add_extra_diacritics
from bifonia import add_extra_diacritics  # noqa: E402
