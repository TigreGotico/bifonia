"""Rebuild bifonia/data/infopedia_pt.csv keyed on bifonia's (word,sense,pos)
slugs. Each sense's definitions come from the Infopédia entry whose pronunciation
matches the sense's IPA (noun readings on the word page; verb readings on the
verb-lemma page). Fetches are cached so the assembly is re-runnable."""
import csv, json, sys, os, pathlib, unicodedata
sys.path.insert(0,"/home/miro/AgentWorkspaces/clients/language/pyinfopedia")
from pyinfopedia.client import get_word, _word_url
from pyinfopedia.transport import Transport
ROOT="/home/miro/AgentWorkspaces/ml/portuguese/homogr"
CACHE="/home/miro/AgentWorkspaces/.tmp/infopedia_cache.jsonl"
T=Transport(mode="flaresolverr", flaresolverr_url="http://192.168.1.116:8191")

def nrm(ipa):
    if not ipa: return ""
    s=unicodedata.normalize("NFC", ipa).lower()
    repl={"ˈ":"","ˌ":"","ə":"i","ɨ":"i","ɐ":"a","ɫ":"l","ʀ":"ʁ","r":"ɾ","ɡ":"g",
          "ʧ":"tʃ","/":"","[":"","]":"","(":"",")":"",":":""," ":""}
    for a,b in repl.items(): s=s.replace(a,b)
    return s

# bifonia roster + lemmas
roster=list(csv.DictReader(open(f"{ROOT}/bifonia/data/heterophonic_homographs.csv",encoding="utf-8")))
lemma={}
for r in csv.DictReader(open(f"{ROOT}/bifonia/data/infopedia_pt.csv",encoding="utf-8")):
    if r.get("lemma"): lemma[(r["word"],r["sense"])]=r["lemma"]
lemma[("molho","sauce")]="molhar"
ADJ_LEMMA={("tola","foolish"):"tolo"}
lemma[("renovo","renew")]="renovar"; lemma[("soma","add")]="somar"; lemma[("força","force")]="forçar"  # adjective def lives under masc lemma
words=sorted({r["word"] for r in roster})
lemmas=sorted({lemma[(r["word"],r["sense"])] for r in roster
               if "VERB" in r["pos"].split("|") and (r["word"],r["sense"]) in lemma})
extra=open("/home/miro/AgentWorkspaces/.tmp/extra_lemmas.txt").read().split() if os.path.exists("/home/miro/AgentWorkspaces/.tmp/extra_lemmas.txt") else []
fetch_list=sorted(set(words)|set(lemmas)|set(extra))

# cache
cache={}
if os.path.exists(CACHE):
    for l in open(CACHE,encoding="utf-8"):
        if l.strip(): d=json.loads(l); cache[d["__key"]]=d
def fetch(name):
    if name in cache: return cache[name]
    try: e=get_word(name, transport=T)
    except Exception as ex: print(f"  ERR {name}: {ex}",flush=True); e=None
    d=e.to_dict() if e else {"word":name,"categories":[]}
    d["__key"]=name; d["__relations"]=getattr(e,"relations",{}) if e else {}
    with open(CACHE,"a",encoding="utf-8") as fh: fh.write(json.dumps(d,ensure_ascii=False)+"\n")
    cache[name]=d; return d

if "--fetch" in sys.argv:
    for i,name in enumerate(fetch_list):
        if name not in cache:
            fetch(name); print(f"[{i+1}/{len(fetch_list)}] {name}",flush=True)
    print(f"FETCH DONE: {len(cache)} cached")
    sys.exit()

# ASSEMBLE (run after fetch)
def cats_by_pron(entry, ipa_norm):
    return [c for c in entry.get("categories",[]) if nrm(c.get("pronunciation",""))==ipa_norm]
def defs_of(cats):
    out=[]
    for c in cats:
        for s in c.get("senses",[]):
            d=s.get("definition","").strip()
            if d: out.append(d)
    return out
REL=["sinonimos","traducoes","rimas","vizinhas","parecidas","relacionadas","pesquisa"]
rows=[]
for r in roster:
    w,s,pos,ipa=r["word"],r["sense"],r["pos"],r["ipa"]
    we=fetch(w); ipan=nrm(ipa)
    defs=defs_of(cats_by_pron(we, ipan))           # noun (and same-pron) readings on word page
    if (w,s) in ADJ_LEMMA:                          # ADJ def under a differently-spelled lemma
        le=fetch(ADJ_LEMMA[(w,s)])
        acats=[c for c in le.get("categories",[]) if "adj" in (c.get("pos","")or"").lower()]
        defs += defs_of(acats)
    if "VERB" in pos.split("|"):                    # add verb-lemma definitions
        lem=lemma.get((w,s))
        if lem:
            le=fetch(lem)
            vcats=[c for c in le.get("categories",[]) if "verbo" in (c.get("pos","")or"").lower()]
            defs += defs_of(vcats)
    rel=we.get("__relations",{}) or {}
    rows.append({"word":w,"sense":s,"pos":pos,"ipa":ipa,
                 "lemma":lemma.get((w,s),""),
                 "syllabification":we.get("syllabification",""),
                 "etymology":we.get("etymology",""),
                 "definitions":"; ".join(dict.fromkeys(defs)),   # dedup, keep order
                 "url":_word_url(w),
                 **{k:" | ".join(rel.get(k,[])) for k in REL}})
FIELDS=["word","sense","pos","ipa","lemma","syllabification","etymology","definitions","url"]+REL
out=f"{ROOT}/bifonia/data/infopedia_pt.csv"
with open(out,"w",encoding="utf-8",newline="") as fh:
    wr=csv.DictWriter(fh,fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
miss=[(r['word'],r['sense']) for r in rows if not r['definitions']]
print(f"wrote {len(rows)} rows | no-definition: {len(miss)} {miss[:15]}")
