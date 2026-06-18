(ns evoc.gold-test
  "Regression guards: every EVoC stage reproduces the reference (evoc 0.3.1) on
  committed gold fixtures (test/resources/evoc_gold, 512-d blobs, single-thread).
  Mirrors how EVoC tests itself (per-stage equality) — these are the dev/diff_*
  checks frozen as tests. Regenerate fixtures with dev/gold_dump.py if the
  reference changes."
  (:require [clojure.test :refer [deftest testing is]]
            [evoc.mst :as mst]
            [evoc.tree :as tree]
            [evoc.layers :as L]
            [evoc.embed :as embed]
            [umap.graph :as graph]))

;; ---- minimal descr-aware npy reader (C-order; fixtures on classpath) ----
(defn- bytes-of [name]
  (with-open [in (.openStream (clojure.java.io/resource (str "evoc_gold/" name)))]
    (.readAllBytes in)))
(defn- hdr [^bytes bs]
  (let [bb (doto (java.nio.ByteBuffer/wrap bs) (.order java.nio.ByteOrder/LITTLE_ENDIAN))
        hl (do (.position bb 8) (bit-and (int (.getShort bb)) 0xFFFF))
        s (String. bs 10 hl java.nio.charset.StandardCharsets/US_ASCII)
        descr (second (re-find #"'descr':\s*'([^']+)'" s))
        shape (->> (re-find #"'shape':\s*\(([^)]*)\)" s) second (re-seq #"\d+") (mapv #(Long/parseLong %)))]
    [bb (+ 10 hl) descr shape]))
(defn- shape-of [name] (nth (hdr (bytes-of name)) 3))
(defn- f64 ^doubles [name]   ; handles <f8 and <f4
  (let [bs (bytes-of name) [^java.nio.ByteBuffer bb off descr] (hdr bs)
        eb (if (= descr "<f4") 4 8)
        n (quot (- (alength bs) (long off)) eb) a (double-array n)]
    (.position bb (int off))
    (if (= eb 4)
      (let [fb (.asFloatBuffer bb)] (dotimes [i n] (aset a i (double (.get fb)))))
      (.get (.asDoubleBuffer bb) a))
    a))
(defn- i32 ^ints [name]   ; handles <i4 and <i8
  (let [bs (bytes-of name) [^java.nio.ByteBuffer bb off descr] (hdr bs)
        eb (if (= descr "<i8") 8 4)
        n (quot (- (alength bs) (long off)) eb) a (int-array n)]
    (.position bb (int off))
    (if (= eb 8)
      (let [lb (.asLongBuffer bb)] (dotimes [i n] (aset a i (int (.get lb)))))
      (.get (.asIntBuffer bb) a))
    a))
(defn- i64 ^longs [name]   ; emb_rng_val is <i8, kernel wants long[]
  (let [bs (bytes-of name) [^java.nio.ByteBuffer bb off] (hdr bs)
        n (quot (- (alength bs) (long off)) 8) a (long-array n)]
    (.position bb (int off)) (.get (.asLongBuffer bb) a) a))

(defn- maxd [a b] (reduce max 0.0 (map (fn [x y] (Math/abs (- (double x) (double y)))) a b)))
(defn- ari [a b]
  (let [n (count a) pr (fn [x] (/ (* x (dec x)) 2.0))
        tally (fn [ks] (persistent! (reduce (fn [m k] (assoc! m k (inc (get m k 0)))) (transient {}) ks)))
        sij (reduce + (map pr (vals (tally (map vector a b)))))
        sa (reduce + (map pr (vals (tally a)))) sb (reduce + (map pr (vals (tally b))))
        exp (/ (* sa sb) (pr n)) mx (/ (+ sa sb) 2.0)]
    (if (== mx exp) 1.0 (/ (- sij exp) (- mx exp)))))

(def ^:private n 800)

(deftest mst-matches-reference
  (testing "mutual-reachability MST on the gold embedding matches the reference"
    (let [emb (f64 "embedding.npy") [_ dim] (shape-of "embedding.npy")
          r (mst/mutual-reachability-mst emb n (long dim) 5)
          g (f64 "mst_edges.npy")
          gsum (reduce + 0.0 (map #(aget g (+ (* % 3) 2)) (range (dec n))))]
      (is (= (dec n) (alength ^doubles (:w r))) "n-1 edges")
      (is (< (Math/abs (- (reduce + 0.0 (seq ^doubles (:w r))) gsum)) 1e-3) "MST total weight == reference"))))

(deftest condense-bit-exact-on-gold-linkage
  (testing "condense/extract-leaves/cluster-labels are bit-exact on the gold linkage"
    (let [gl (f64 "linkage.npy")
          lk {:left (int-array (map #(int (aget gl (+ (* % 4) 0))) (range (dec n))))
              :right (int-array (map #(int (aget gl (+ (* % 4) 1))) (range (dec n))))
              :delta (double-array (map #(aget gl (+ (* % 4) 2)) (range (dec n))))
              :size (int-array (map #(int (aget gl (+ (* % 4) 3))) (range (dec n))))}
          ct (tree/condense lk n 5)
          leaves (tree/extract-leaves ct)
          labels (vec (tree/cluster-labels ct leaves n))
          g-base (vec (seq (i32 "base_clusters.npy")))]
      (is (= (alength (i32 "ct_parent.npy")) (alength ^ints (:parent ct))) "condensed edge count")
      (is (= 1.0 (ari labels g-base)) "base labels ARI 1.0"))))

(deftest fuzzy-graph-matches-reference
  (testing "umap.graph on the gold kNN reproduces the reference fuzzy graph"
    (let [k 15 idx (i32 "nn_inds.npy") dst (f64 "nn_dists.npy")
          sig (double-array n) rho (double-array n)
          _ (graph/smooth-knn-dist! dst n k sig rho)
          vals (double-array (* n k))
          _ (graph/membership-strengths! idx dst sig rho n k vals)
          {:keys [head tail weights]} (graph/symmetrize idx vals n k)
          key (fn [i j] (+ (* (long (min i j)) n) (long (max i j))))
          rmap (persistent! (reduce (fn [m e] (assoc! m (key (aget ^ints head e) (aget ^ints tail e)) (aget ^doubles weights e))) (transient {}) (range (alength ^doubles weights))))
          gr (i32 "graph_row.npy") gc (i32 "graph_col.npy") gd (f64 "graph_data.npy")
          gmap (persistent! (reduce (fn [m e] (let [i (aget gr e) j (aget gc e)] (if (= i j) m (assoc! m (key i j) (aget gd e))))) (transient {}) (range (alength gd))))]
      (is (= (count rmap) (count gmap)) "same undirected edge count")
      (is (< (reduce (fn [mx kk] (max mx (Math/abs (- (double (get rmap kk 0.0)) (double (get gmap kk 0.0)))))) 0.0 (keys gmap)) 1e-5) "weights match"))))

(deftest embedding-kernel-matches-reference
  (testing "reproducible embedding kernel matches the reference after 1 real epoch (2 epochs)"
    (let [meta (i32 "emb_meta.npy") dim (aget meta 1) V (aget meta 2) block (aget meta 3)
          init (f64 "init_embedding.npy") emb (java.util.Arrays/copyOf init (alength init))
          eps (f64 "emb_eps.npy") epn (f64 "emb_epn.npy")
          eons (java.util.Arrays/copyOf eps (alength eps)) eonsn (java.util.Arrays/copyOf epn (alength epn))
          upd (double-array (* V dim))]
      (embed/node-embedding-layout-repr! emb (i32 "csr_indptr.npy") (i32 "csr_indices.npy")
                                         eps epn eons eonsn (i64 "emb_rng_val.npy")
                                         (i32 "emb_node_order.npy") (f64 "emb_gamma.npy") upd
                                         0.5 0.1 (long V) (long dim) 2 (long block))
      (is (< (maxd (seq emb) (seq (f64 "embedding_repr_after2.npy"))) 1e-4) "embedding after 1 real epoch matches"))))

(deftest membership-and-multires-match-reference
  (testing "membership strengths + multi-resolution layers match the reference"
    (let [ct {:parent (i32 "ct_parent.npy") :child (i32 "ct_child.npy")
              :lambda (f64 "ct_lambda.npy") :size (i32 "ct_child_size.npy")}
          leaves (i32 "leaves.npy")
          labels (tree/cluster-label-vector ct leaves 0.0 n)
          strengths (tree/point-membership-strengths ct leaves labels n)
          cluster-tree (tree/mask-condensed-tree ct n)
          bc (L/min-cluster-size-barcode cluster-tree n 5)
          tpr (L/compute-total-persistence bc)
          peaks (L/find-peaks (:total-persistence tpr))
          sel (L/select-diverse-peaks peaks (:total-persistence tpr) (:sizes tpr) (:births bc) (:deaths bc) 0.2 9)]
      (is (< (maxd (seq strengths) (seq (f64 "base_strengths.npy"))) 1e-5) "membership strengths match")
      (is (< (maxd (seq (:births bc)) (seq (f64 "bc_births.npy"))) 1e-5) "barcode births match")
      (is (< (maxd (seq (:total-persistence tpr)) (seq (f64 "total_persistence.npy"))) 1e-4) "total persistence matches")
      (is (= (vec (seq peaks)) (vec (seq (i32 "peaks.npy")))) "peaks match")
      (is (= (set (seq sel)) (set (seq (i32 "selected_peaks.npy")))) "selected peaks match"))))
