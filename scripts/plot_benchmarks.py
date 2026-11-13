import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT="docs/img"
BLUE="#2c6fbb"; GREY="#b8c4d0"; GREEN="#2e8b57"; ORANGE="#e08a1e"

# 1) OOD accuracy by approach
approaches=[("majority sense*",47.4,GREY),("Naive-Bayes",84.7,GREY),("perceptron",85.6,GREY),
            ("rules\n(zero-dep)",89.8,BLUE),("spaCy lg\n(POS→sense)",91.6,ORANGE),
            ("hybrid ensemble\n(POS⊕rules)",95.3,GREEN)]
fig,ax=plt.subplots(figsize=(8,4.5))
names=[a[0] for a in approaches]; vals=[a[1] for a in approaches]; cols=[a[2] for a in approaches]
bars=ax.bar(names,vals,color=cols,edgecolor="white")
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.6,f"{v:.1f}",ha="center",fontsize=10,fontweight="bold")
ax.set_ylim(0,100); ax.set_ylabel("Accuracy (%)")
ax.set_title("Out-of-distribution accuracy — 7,119 real sentences (112 words)")
ax.spines[["top","right"]].set_visible(False)
ax.text(0,-14,"* corpus-derived majority sense; wild dominant-sense baseline ≈ 75%",fontsize=7,color="#666")
plt.tight_layout(); plt.savefig(f"{OUT}/ood_accuracy.png",dpi=140); plt.close()

# 2) ensemble breakdown by subset
subsets=["POS-separable\n(n=6583)","POS-ambiguous\n(n=536)","all\n(n=7119)"]
spacy=[94.8,52.6,91.6]; rules=[89.9,89.7,89.9]; ens=[95.8,89.9,95.3]
import numpy as np
x=np.arange(len(subsets)); w=0.26
fig,ax=plt.subplots(figsize=(8,4.5))
ax.bar(x-w,spacy,w,label="spaCy (POS→sense)",color=ORANGE)
ax.bar(x,rules,w,label="rules",color=BLUE)
ax.bar(x+w,ens,w,label="hybrid ensemble",color=GREEN)
for i,(a,b,c) in enumerate(zip(spacy,rules,ens)):
    for off,v in [(-w,a),(0,b),(w,c)]: ax.text(i+off,v+1,f"{v:.0f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(subsets); ax.set_ylim(0,105); ax.set_ylabel("Accuracy (%)")
ax.set_title("Where each approach wins — POS-separable vs same-POS readings")
ax.legend(frameon=False,loc="lower left"); ax.spines[["top","right"]].set_visible(False)
ax.text(1,40,"POS tagging is at chance\non same-POS pairs\n(sede, molho, corte, forma,\nbola, cor, lobo, polo, tola)",ha="center",fontsize=7.5,color="#444")
plt.tight_layout(); plt.savefig(f"{OUT}/ensemble_breakdown.png",dpi=140); plt.close()

# 3) balanced hard subset (same-POS words, equal senses)
words=["sede","forma","molho","gosto","corte","gozo","coro","posto"]
rules_bh=[88,78,94,99,100,98,99,94]; spacy_bh=[50,50,50,55,72,50,50,68]
fig,ax=plt.subplots(figsize=(8,4))
x=np.arange(len(words))
ax.bar(x-0.2,spacy_bh,0.4,label="spaCy",color=ORANGE)
ax.bar(x+0.2,rules_bh,0.4,label="bifonia rules",color=BLUE)
ax.set_xticks(x); ax.set_xticklabels(words); ax.set_ylim(0,105); ax.set_ylabel("Accuracy (%)")
ax.set_title("Balanced hard subset (equal senses) — meaning beats POS")
ax.axhline(50,ls="--",lw=0.8,color="#aaa"); ax.legend(frameon=False)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/balanced_hard.png",dpi=140); plt.close()
print("wrote ood_accuracy.png, ensemble_breakdown.png, balanced_hard.png")
