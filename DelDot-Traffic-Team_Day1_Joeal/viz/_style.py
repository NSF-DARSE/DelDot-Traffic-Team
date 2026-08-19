import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "outputs"
INK="#111a22"; SIGN="#0a6b4e"; AMBER="#e8a200"; RED="#cf2f3f"; BLUE="#378add"; GREY="#8496a6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":"#d3dbe3",
    "axes.grid":True,"grid.color":"#eef1f4","axes.axisbelow":True,"figure.dpi":150,
    "axes.titleweight":"bold","axes.titlesize":13,"axes.titlecolor":INK})
def load():
    z=np.load(OUT/"artifacts.npz",allow_pickle=True)
    m=json.load(open(OUT/"metrics.json"))
    return z,m
def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/"figures"/name,bbox_inches="tight"); plt.close(fig)
    print("saved",name)
