import json
from pathlib import Path
PD = Path(".")
BBS = ["vit","clip","dinov3","rad_dino","dinov2","mae","beit"]
def get(bb,m,ds,key):
    p = PD/"outputs"/bb/"seed_42"/"results_core3"/m/"results.json"
    return json.loads(p.read_text()).get(ds,{}).get(key) if p.exists() else None
def f(v): return f"{v:.3f}" if isinstance(v,(int,float)) else " -- "
print(f"{'backbone':10}{'ISIC g|na':>16}{'CheX g|na':>16}{'Path g|na':>16}")
for bb in BBS:
    i1,i2=get(bb,'gram_ls','isic2017','balanced_accuracy'),get(bb,'gram_ls_na','isic2017','balanced_accuracy')
    c1,c2=get(bb,'gram_ls','chexpert','macro_auroc'),get(bb,'gram_ls_na','chexpert','macro_auroc')
    p1,p2=get(bb,'gram_ls','pathmnist','balanced_accuracy'),get(bb,'gram_ls_na','pathmnist','balanced_accuracy')
    print(f"{bb:10}{f(i1)+'|'+f(i2):>16}{f(c1)+'|'+f(c2):>16}{f(p1)+'|'+f(p2):>16}")
