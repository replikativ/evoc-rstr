"""EVoC demo: reference clustering + a 2-D UMAP layout for visualization.

Takes an MNIST subset, runs reference evoc.EVoC (the gold clustering), and a 2-D
umap-learn layout (purely for plotting — EVoC itself clusters a 4-D embedding).
Dumps X/y + reference labels + viz coords so the Clojure side runs evoc-rstr on the
identical data and we plot all three side by side (truth | reference | evoc-rstr).

    python3 dev/evoc_gen.py [N]
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from evoc.clustering import EVoC
import umap

OUT = "/tmp/evoc_demo"; os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SEED = 42
rng = np.random.RandomState(SEED)

X_all = np.load("/tmp/umap_gold/mnist_X.npy")
y_all = np.load("/tmp/umap_gold/mnist_y.npy")
idx = rng.choice(len(X_all), N, replace=False)
X = np.ascontiguousarray(X_all[idx].astype(np.float64))
y = np.ascontiguousarray(y_all[idx].astype(np.int32))
np.save(f"{OUT}/X.npy", X); np.save(f"{OUT}/y.npy", y)
print(f"[evoc-gen] X {X.shape} y {y.shape}", flush=True)

# reference EVoC clustering
t0 = time.perf_counter()
ref = EVoC(random_state=np.random.RandomState(SEED)).fit_predict(X.astype(np.float32))
ref_secs = time.perf_counter() - t0
np.save(f"{OUT}/ref_labels.npy", np.ascontiguousarray(ref.astype(np.int32)))
print(f"[evoc-gen] reference EVoC: {ref_secs:.1f}s  clusters={int(ref.max()+1)} "
      f"noise={int((ref<0).sum())}  ARI-vs-truth={ari(y, ref):.3f}", flush=True)

# 2-D UMAP layout for visualization only
t0 = time.perf_counter()
viz = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                random_state=SEED).fit_transform(X.astype(np.float32))
np.save(f"{OUT}/viz_2d.npy", np.ascontiguousarray(viz.astype(np.float64)))
print(f"[evoc-gen] 2-D umap viz: {time.perf_counter()-t0:.1f}s -> {OUT}/{{X,y,ref_labels,viz_2d}}.npy")
