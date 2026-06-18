(ns evoc
  "EVoC clustering orchestrator (ports clustering.evoc_clusters + EVoC.fit_predict).
  Pipeline: cosine kNN -> fuzzy simplicial set (umap-rstr) -> node embedding ->
  multi-resolution cluster layers; labels_ = argmax(persistence).

  The numeric kernels are deftm (embedding, barcode, persistence, MST/tree, graph)
  so they devirtualize / fit raster's compile path; only the wiring is plain defn."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= ==])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= ==]]
            [raster.spatial.nndescent :as nnd]
            [umap.graph :as graph]
            [evoc.embed :as embed]
            [evoc.layers :as layers]))

;; EVoC make_epochs_per_sample: eps[i] = n_epochs / max(n_epochs*w/wmax, 1).
(deftm make-epochs-per-sample!
  [weights :- (Array double) m :- Long n-epochs :- Long eps :- (Array double)] :- (Array double)
  (let [wmax (loop [i 0 mx 0.0]
               (if (< i m) (recur (inc i) (if (> (aget weights i) mx) (aget weights i) mx)) mx))]
    (dotimes [i m]
      (let [s (* (double n-epochs) (/ (aget weights i) wmax))
            s (if (< s 1.0) 1.0 s)]
        (aset eps i (/ (double n-epochs) s)))))
  eps)

;; COO (head,tail,weights) -> CSR (indptr,indices,data) grouped by head (counting sort).
(deftm coo->csr!
  [head :- (Array int) tail :- (Array int) weights :- (Array double) n :- Long m :- Long
   indptr :- (Array int) indices :- (Array int) data :- (Array double) cursor :- (Array int)]
  :- (Array int)
  (dotimes [i (+ n 1)] (aset indptr i (int 0)))
  (dotimes [e m] (let [h (+ (aget head e) 1)] (aset indptr h (int (+ (aget indptr h) 1)))))
  (dotimes [i n] (aset indptr (+ i 1) (int (+ (aget indptr (+ i 1)) (aget indptr i)))))
  (dotimes [i n] (aset cursor i (aget indptr i)))
  (dotimes [e m]
    (let [h (long (aget head e)) pos (long (aget cursor h))]
      (aset indices pos (aget tail e)) (aset data pos (aget weights e))
      (aset cursor h (int (+ pos 1)))))
  indptr)

;; scale a double[] by c into a fresh array (epn = eps/neg_rate*1.5).
(deftm scale-into [a :- (Array double) m :- Long c :- Double out :- (Array double)] :- (Array double)
  (dotimes [i m] (aset out i (* c (aget a i)))) out)

;; ---- thin orchestration (plain defn) ----

(defn- dcopy ^doubles [^doubles x] (java.util.Arrays/copyOf x (clojure.core/alength x)))

(defn- random-init
  "Deterministic random normal(scale=0.25) init — EVoC's node_embedding_init=None.
  (PCA base_init hits a native BLAS dgemm crash and the init is washed out by the
  contractive SGD anyway, so this is both safe and a faithful EVoC config.)"
  ^doubles [n n-comp seed]
  (let [a (double-array (clojure.core/* (long n) (long n-comp)))
        rng (java.util.Random. (long seed))]
    (dotimes [i (clojure.core/alength a)] (clojure.core/aset a i (clojure.core/* 0.25 (.nextGaussian rng))))
    a))

(defn- graph-csr
  "cosine kNN -> fuzzy simplicial set -> symmetrize -> CSR (grouped by head)."
  [^doubles X n dim k]
  (let [{:keys [idx dst]} (nnd/cosine-knn (dcopy X) n dim k)   ; cosine-knn mutates X -> copy
        sigmas (double-array n) rhos (double-array n)
        _ (graph/smooth-knn-dist! dst (long n) (long k) sigmas rhos)
        vals (double-array (clojure.core/* (long n) (long k)))
        _ (graph/membership-strengths! idx dst sigmas rhos (long n) (long k) vals)
        {:keys [head tail weights]} (graph/symmetrize idx vals (long n) (long k))
        m (clojure.core/alength ^doubles weights)
        indptr (int-array (clojure.core/inc (long n))) indices (int-array m)
        data (double-array m) cursor (int-array n)]
    (coo->csr! head tail weights (long n) (long m) indptr indices data cursor)
    {:indptr indptr :indices indices :data data :m m}))

(defn- shuffle-into! [^ints order ^java.util.Random rng]
  ;; in-place Fisher-Yates (matches np.random.shuffle direction not required —
  ;; we only need determinism + a valid permutation)
  (loop [i (clojure.core/dec (clojure.core/alength order))]
    (when (clojure.core/pos? i)
      (let [j (.nextInt rng (clojure.core/inc i))
            t (clojure.core/aget order i)]
        (clojure.core/aset order i (clojure.core/aget order j))
        (clojure.core/aset order j t))
      (recur (clojure.core/dec i)))))

(defn- run-embedding
  "Run the reproducible node-embedding driver over the CSR graph from `init`."
  ^doubles [csr ^doubles init n n-comp n-epochs noise seed]
  (let [{:keys [^ints indptr ^ints indices ^doubles data m]} csr
        eps (make-epochs-per-sample! data (long m) (long n-epochs) (double-array m))
        epn (scale-into eps (long m) 1.5 (double-array m))     ; neg_rate=1.0, reproducible *1.5
        eons (dcopy eps) eonsn (dcopy epn)
        rng (java.util.Random. (long seed))
        rng-vals (long-array n-epochs)
        node-order (int-array (clojure.core/* (long n-epochs) (long n)))
        gamma (double-array n-epochs)
        order (int-array n)
        block-size (clojure.core/max 1024 (clojure.core/quot (long n) 8))
        emb (dcopy init) updates (double-array (clojure.core/* (long n) (long n-comp)))]
    (dotimes [i n] (clojure.core/aset order i (int i)))
    (dotimes [ep n-epochs]
      (clojure.core/aset rng-vals ep (long (.nextInt rng Integer/MAX_VALUE)))
      (clojure.core/aset gamma ep (double (clojure.core/+ 0.5 (clojure.core/* 1.0 (clojure.core// (double ep) (clojure.core/max 1 (clojure.core/dec (long n-epochs))))))))
      (System/arraycopy order 0 node-order (clojure.core/* ep (long n)) (long n))
      (shuffle-into! order rng))
    (embed/node-embedding-layout-repr! emb indptr indices eps epn eons eonsn
                                       rng-vals node-order gamma updates
                                       (double noise) 0.1 (long n) (long n-comp) (long n-epochs) (long block-size))
    emb))

(defn fit-predict
  "Cluster X (flat row-major double[n*dim]). Returns {:labels int[n] (noise=-1)
  :persistence [..] :layers [int[]] :embedding double[n*out-dim]}. Mirrors
  evoc.EVoC().fit_predict with a deterministic PCA init."
  [X n dim & {:keys [k n-epochs min-samples base-min-cluster-size noise seed max-layers min-similarity]
              :or {k 15 n-epochs 50 min-samples 5 base-min-cluster-size 5 noise 0.5 seed 42
                   max-layers 10 min-similarity 0.2}}]
  (let [n (long n) dim (long dim)
        X (let [^doubles d (if (clojure.core/instance? (Class/forName "[D") X) X
                               (let [a (double-array (clojure.core/alength ^floats X))]
                                 (dotimes [i (clojure.core/alength a)] (clojure.core/aset a i (double (clojure.core/aget ^floats X i)))) a))] d)
        n-comp (clojure.core/min (clojure.core/max (clojure.core/quot (long k) 4) 4) 15)
        csr (graph-csr X n dim k)
        init (random-init n n-comp seed)
        emb (run-embedding csr init n n-comp n-epochs noise seed)
        res (layers/build-cluster-layers emb n n-comp
                                         {:min-samples min-samples :base-min-cluster-size base-min-cluster-size
                                          :min-similarity min-similarity :max-layers max-layers})
        persistence (:persistence res)
        best (first (apply max-key second (map-indexed vector persistence)))
        labels (nth (:layers res) best)]
    {:labels labels :persistence persistence :layers (:layers res)
     :strengths (nth (:strengths res) best) :embedding emb :n-clusters (clojure.core/inc (reduce clojure.core/max -1 (seq ^ints labels)))}))
