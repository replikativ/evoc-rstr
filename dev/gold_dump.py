"""Per-stage gold dump for the EVoC port (evoc-rstr).

Drives the reference `evoc` (0.3.1, ~/Development/evoc) pipeline stage by stage on a
FIXED dataset + RandomState(42), dumping every intermediate so the Clojure side can
feed identical inputs to each `evoc` stage and diff the output. This is the
measurement backbone — match the reference per stage, not just end-to-end.

Mirrors evoc.clustering.evoc_clusters + build_cluster_layers with the default
(seeded) config. Dumps to /tmp/evoc_gold/. Run with the env that has `evoc`:
    python3 dev/gold_dump.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_blobs

from evoc.knn_graph import knn_graph
from evoc.graph_construction import neighbor_graph_matrix
from evoc.label_propagation import label_propagation_init
from evoc.node_embedding import node_embedding
from evoc.numba_kdtree import build_kdtree
from evoc.boruvka import parallel_boruvka
from evoc.cluster_trees import (
    mst_to_linkage_tree, condense_tree, extract_leaves,
    get_cluster_label_vector, get_point_membership_strength_vector,
    mask_condensed_tree,
)
from evoc.clustering_utilities import (
    min_cluster_size_barcode, compute_total_persistence, find_peaks,
    select_diverse_peaks, extract_clusters_by_id,
)
from evoc.clustering import EVoC
import numba

OUT = "/tmp/evoc_gold"; os.makedirs(OUT, exist_ok=True)
def save(name, arr): np.save(f"{OUT}/{name}.npy", np.ascontiguousarray(arr))

# ---- fixed dataset (cosine on float; positive-ish box so cosine is well-defined) ----
SEED = 42
# EVoC's own test regime (test_clustering.py): high-dim blobs. EVoC is built for
# high-dim embedding-like data; on low-dim it clusters poorly/unstably.
X, y_true = make_blobs(n_samples=800, n_features=512, centers=4,
                       cluster_std=0.8, random_state=SEED)
X = np.ascontiguousarray(X.astype(np.float32))
save("X", X); save("y_true", y_true.astype(np.int32))
n_samples = X.shape[0]

# default EVoC params
n_neighbors, min_samples, n_epochs = 15, 5, 50
noise_level, base_min_cluster_size = 0.5, 5
neighbor_scale, min_similarity_threshold, max_layers, n_label_prop_iter = 1.0, 0.2, 10, 20
n_components = min(max(n_neighbors // 4, 4), 15)   # = 4
save("params", np.array([n_neighbors, min_samples, n_epochs, base_min_cluster_size,
                         n_components, max_layers], dtype=np.int32))

rs = np.random.RandomState(SEED)

# ---- stage 1: kNN graph ----
nn_inds, nn_dists = knn_graph(X, n_neighbors=n_neighbors, random_state=rs)
save("nn_inds", nn_inds.astype(np.int32)); save("nn_dists", nn_dists.astype(np.float32))

# ---- stage 2: fuzzy graph (COO) ----
graph = neighbor_graph_matrix(neighbor_scale * n_neighbors, nn_inds, nn_dists, True)
g = graph.tocoo()
save("graph_row", g.row.astype(np.int32)); save("graph_col", g.col.astype(np.int32))
save("graph_data", g.data.astype(np.float32)); save("graph_shape", np.array(graph.shape, np.int32))

# ---- stage 3: label-prop init ----
init_embedding = label_propagation_init(
    graph, n_components=n_components,
    approx_n_parts=int(np.clip(int(8 * np.sqrt(n_samples)), 256, 16384)),
    random_scale=0.1, scaling=0.5, noise_level=noise_level,
    random_state=rs, data=X, n_label_prop_iter=n_label_prop_iter)
save("init_embedding", init_embedding.astype(np.float32))

# ---- stage 4: node embedding (SGD) ----
embedding = node_embedding(
    graph, n_components=n_components, n_epochs=n_epochs,
    initial_embedding=init_embedding, negative_sample_rate=1.0,
    noise_level=noise_level, random_state=rs, verbose=False,
    reproducible_flag=True, initial_alpha=0.1)
save("embedding", embedding.astype(np.float32))

# ---- stage 4b: reproducible-embedding gold with EXPLICIT dumped inputs ----
# Drives node_embedding_epoch_repr directly (replicating the node_embedding driver)
# from the gold init, with a dedicated RNG, so raster can be fed identical inputs and
# reproduce the embedding deterministically. Dumps after epoch 0 (1-epoch determinism
# check) and after all epochs.
from evoc.node_embedding import node_embedding_epoch_repr, make_epochs_per_sample
INT32_MAX = 2 ** 31 - 1
csr = graph.tocsr()
save("csr_indptr", csr.indptr.astype(np.int32)); save("csr_indices", csr.indices.astype(np.int32))
eps_g = make_epochs_per_sample(csr.data, n_epochs).astype(np.float32, order="C")
epn_g = (eps_g / 1.0) * 1.5            # negative_sample_rate=1.0, reproducible -> *1.5
save("emb_eps", eps_g); save("emb_epn", epn_g)
rs_e = np.random.RandomState(12345)
rng_val = rs_e.randint(INT32_MAX, size=n_epochs).astype(np.int64)
gamma_sched = np.linspace(0.5, 1.5, n_epochs).astype(np.float64)
save("emb_rng_val", rng_val); save("emb_gamma", gamma_sched)
V = int(graph.shape[0]); block_size = max(1024, V // 8)
save("emb_meta", np.array([n_epochs, n_components, V, block_size], np.int32))
# NB: stage-4 node_embedding() mutates init_embedding in place (no copy), so reload
# the clean label-prop init from disk rather than the now-corrupted variable.
emb_r = np.load(f"{OUT}/init_embedding.npy").astype(np.float32, order="C").copy()
eons_g = eps_g.copy(); eonns_g = epn_g.copy()
updates = np.zeros_like(emb_r)
node_order = np.arange(V, dtype=np.uint32)
no_per_epoch = np.zeros((n_epochs, V), np.int32)
indptr_u4 = csr.indptr.astype(np.uint32); indices_u4 = csr.indices.astype(np.uint32)
alpha = np.float32(0.1)
emb_after1 = None
for nn in range(n_epochs):
    no_per_epoch[nn] = node_order
    node_embedding_epoch_repr(emb_r, indptr_u4, indices_u4, np.uint32(V), eps_g,
                              np.uint32(rng_val[nn]), np.uint8(n_components), alpha,
                              epn_g, eonns_g, eons_g, np.uint8(nn), np.float32(noise_level),
                              gamma_sched[nn], updates, node_order, np.uint32(block_size))
    if nn == 0:
        emb_after1 = emb_r.copy()
    if nn == 1:
        save("embedding_repr_after2", emb_r.copy())
    if nn == 4:
        save("embedding_repr_after5", emb_r.copy())
    updates *= (1.0 - alpha) ** 2 * 0.5
    rs_e.shuffle(node_order)
    alpha = np.float32(0.1 * (1.0 - nn / n_epochs))
save("emb_node_order", no_per_epoch)
save("embedding_repr_after1", emb_after1)
save("embedding_repr", emb_r)
print(f"[gold] repr-embedding: block_size={block_size} V={V} dim={n_components} nnz={len(csr.data)}")

# ---- stage 5: build_cluster_layers internals (replicated body) ----
n_threads = numba.get_num_threads()
numba_tree = build_kdtree(embedding.astype(np.float32))
edges = parallel_boruvka(numba_tree, n_threads, min_samples=min_samples, reproducible=True)
save("mst_edges", edges)
sorted_mst = edges[np.argsort(edges.T[2])]
save("sorted_mst", sorted_mst)
uncondensed = mst_to_linkage_tree(sorted_mst)
save("linkage", np.asarray(uncondensed))
ct = condense_tree(uncondensed, base_min_cluster_size)
save("ct_parent", ct.parent.astype(np.int32)); save("ct_child", ct.child.astype(np.int32))
save("ct_lambda", ct.lambda_val.astype(np.float64)); save("ct_child_size", ct.child_size.astype(np.int32))
leaves = extract_leaves(ct)
save("leaves", np.asarray(leaves).astype(np.int32))
base_clusters = get_cluster_label_vector(ct, leaves, 0.0, n_samples)
save("base_clusters", base_clusters.astype(np.int32))
base_strengths = get_point_membership_strength_vector(ct, leaves, base_clusters)
save("base_strengths", base_strengths.astype(np.float32))

mask = ct.child >= n_samples
cluster_tree = mask_condensed_tree(ct, mask)
births, deaths, parents, lambda_deaths = min_cluster_size_barcode(cluster_tree, n_samples, base_min_cluster_size)
save("bc_births", np.asarray(births)); save("bc_deaths", np.asarray(deaths))
save("bc_parents", np.asarray(parents)); save("bc_lambda_deaths", np.asarray(lambda_deaths))
sizes, total_persistence = compute_total_persistence(births, deaths, lambda_deaths)
save("sizes", np.asarray(sizes)); save("total_persistence", np.asarray(total_persistence))
peaks = find_peaks(total_persistence)
save("peaks", np.asarray(peaks).astype(np.int32))
selected_peaks = select_diverse_peaks(peaks, total_persistence, sizes, births, deaths,
                                      min_similarity_threshold=min_similarity_threshold,
                                      max_layers=max_layers - 1)
save("selected_peaks", np.asarray(selected_peaks).astype(np.int32))

# ---- stage 6: full pipeline labels (the deliverable) ----
model = EVoC(random_state=np.random.RandomState(SEED))
labels = model.fit_predict(X)
save("final_labels", labels.astype(np.int32))
save("persistence_scores", np.asarray(model.persistence_scores_))
for i, layer in enumerate(model.cluster_layers_):
    save(f"layer_{i}", np.asarray(layer).astype(np.int32))

print(f"[gold] X {X.shape}  knn {nn_inds.shape}  graph nnz {len(g.data)}  emb {embedding.shape}")
print(f"[gold] condensed-tree edges {len(ct.parent)}  leaves {len(leaves)}  peaks {len(peaks)} selected {len(selected_peaks)}")
print(f"[gold] cluster_layers {len(model.cluster_layers_)}  final clusters {labels.max()+1} (noise {int((labels<0).sum())})")
print(f"[gold] persistence_scores {np.asarray(model.persistence_scores_)}")
print(f"[gold] wrote -> {OUT}")
