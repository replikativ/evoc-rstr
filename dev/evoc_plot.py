"""Plot the EVoC comparison: 2-D UMAP layout colored by true label | reference
EVoC clusters | evoc-rstr clusters, with ARI metrics. Mirrors the UMAP harness."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari

D = "/tmp/evoc_demo"
viz = np.load(f"{D}/viz_2d.npy"); y = np.load(f"{D}/y.npy")
ref = np.load(f"{D}/ref_labels.npy")
rstr = np.rint(np.load(f"{D}/rstr_labels.npy").ravel()).astype(int)

def panel(ax, labels, title):
    noise = labels < 0
    # stable color per cluster id
    ax.scatter(viz[noise,0], viz[noise,1], c="lightgray", s=3, linewidths=0)
    pts = ~noise
    ax.scatter(viz[pts,0], viz[pts,1], c=labels[pts] % 20, cmap="tab20", s=3, linewidths=0)
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])

ari_rr = ari(rstr, ref); ari_rt = ari(rstr, y); ari_reft = ari(ref, y)
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
panel(ax[0], y, f"true labels ({len(set(y))} classes)")
panel(ax[1], ref, f"reference EVoC — {int(ref.max()+1)} clusters, {int((ref<0).mean()*100)}% noise\nARI-vs-truth {ari_reft:.3f}")
panel(ax[2], rstr, f"evoc-rstr — {int(rstr.max()+1)} clusters, {int((rstr<0).mean()*100)}% noise\nARI-vs-truth {ari_rt:.3f}")
fig.suptitle(f"EVoC clustering on MNIST (n={len(y)}), 2-D UMAP layout  |  ARI(evoc-rstr, reference)={ari_rr:.3f}", fontsize=13)
fig.tight_layout()
fig.savefig(f"{D}/evoc_compare.png", dpi=110); plt.close(fig)
print(f"wrote {D}/evoc_compare.png")
print(f"ARI(evoc-rstr, reference EVoC) = {ari_rr:.3f}")
print(f"ARI vs truth: reference {ari_reft:.3f}  evoc-rstr {ari_rt:.3f}")
print(f"clusters: reference {int(ref.max()+1)} (noise {int((ref<0).mean()*100)}%)  "
      f"evoc-rstr {int(rstr.max()+1)} (noise {int((rstr<0).mean()*100)}%)")
