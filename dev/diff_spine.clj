(ns diff-spine
  "Per-stage differential: run raster's clustering spine on the GOLD embedding and
  compare each stage (MST, linkage, condensed tree, leaves, base labels) to the
  reference dump in /tmp/evoc_gold. Run from a classpath with raster + evoc-rstr."
  (:require [evoc.mst :as mst]
            [evoc.tree :as tree]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def G "/tmp/evoc_gold")
(defn col [m i ncol] (let [d (:data m)] (mapv #(aget d (+ (* % ncol) i)) (range (quot (alength d) ncol)))))

;; adjusted Rand index between two int label vectors
(defn ari [a b]
  (let [n (count a)
        pairs (fn [x] (/ (* x (dec x)) 2.0))
        cont (reduce (fn [m i] (update m [(nth a i) (nth b i)] (fnil inc 0))) {} (range n))
        ai (reduce (fn [m i] (update m (nth a i) (fnil inc 0))) {} (range n))
        bi (reduce (fn [m i] (update m (nth b i) (fnil inc 0))) {} (range n))
        sij (reduce + (map (fn [[_ v]] (pairs v)) cont))
        sa (reduce + (map (fn [[_ v]] (pairs v)) ai))
        sb (reduce + (map (fn [[_ v]] (pairs v)) bi))
        exp (/ (* sa sb) (pairs n))
        mx (/ (+ sa sb) 2.0)]
    (if (== mx exp) 1.0 (/ (- sij exp) (- mx exp)))))

(let [embm (npy/load-npy (str G "/embedding.npy"))
      [n dim] (:shape embm)
      n (long n) dim (long dim)
      emb (npy/read-f64 (str G "/embedding.npy"))
      ms 5
      ;; --- gold ---
      g-mst (npy/load-npy (str G "/mst_edges.npy"))            ; (n-1, 3) from,to,weight
      g-mst-w (col g-mst 2 3)
      g-link (npy/load-npy (str G "/linkage.npy"))
      g-ct-parent (npy/read-i32 (str G "/ct_parent.npy"))
      g-leaves (npy/read-i32 (str G "/leaves.npy"))
      g-base (npy/read-i32 (str G "/base_clusters.npy"))
      ;; --- raster spine ---
      r (mst/mutual-reachability-mst emb n dim ms)
      s (mst/sort-edges r)
      r-w (vec (seq ^doubles (:w s)))
      lk (tree/linkage-tree (:from s) (:to s) (:w s) n)
      ct (tree/condense lk n 5)
      leaves (tree/extract-leaves ct)
      labels (vec (tree/cluster-labels ct leaves n))
      ;; --- ISOLATION: run raster condense on the GOLD linkage (rules out MST/linkage f32/f64) ---
      gl (npy/load-npy (str G "/linkage.npy"))
      gld (:data gl)
      lk-gold {:left  (int-array (map #(int (aget gld (+ (* % 4) 0))) (range (dec n))))
               :right (int-array (map #(int (aget gld (+ (* % 4) 1))) (range (dec n))))
               :delta (double-array (map #(aget gld (+ (* % 4) 2)) (range (dec n))))
               :size  (int-array (map #(int (aget gld (+ (* % 4) 3))) (range (dec n))))}
      ct-g (tree/condense lk-gold n 5)
      leaves-g (tree/extract-leaves ct-g)
      labels-g (vec (tree/cluster-labels ct-g leaves-g n))
      g-ct-lambda (npy/read-f64 (str G "/ct_lambda.npy"))
      sum (fn [xs] (reduce + 0.0 xs))
      out (str
            (format "n=%d dim=%d  min_samples=%d\n" n dim ms)
            (format "MST   : raster edges=%d  gold edges=%d\n" (count r-w) (count g-mst-w))
            (format "        raster sum|w|=%.4f  gold sum|w|=%.4f  raster sum w^2=%.4f\n"
                    (sum r-w) (sum g-mst-w) (sum (map #(* % %) r-w)))
            (format "LINK  : raster rows=%d  gold rows=%d  shape=%s\n"
                    (alength ^doubles (:delta lk)) (first (:shape g-link)) (:shape g-link))
            (format "CONDS : raster edges=%d  gold edges=%d\n" (alength ^ints (:parent ct)) (alength g-ct-parent))
            (format "LEAVES: raster=%d  gold=%d  set-eq=%s\n"
                    (count leaves) (count g-leaves) (= (set leaves) (set (seq g-leaves))))
            (format "BASE  : raster nclusters=%d (noise %d)  gold nclusters=%d (noise %d)\n"
                    (count (distinct (filter #(>= % 0) labels))) (count (filter neg? labels))
                    (count (distinct (filter #(>= % 0) (seq g-base)))) (count (filter neg? (seq g-base))))
            (format "BASE  : ARI(raster, gold)=%.4f\n" (ari labels (vec (seq g-base))))
            "--- isolation: raster condense on GOLD linkage (rules out MST/linkage f32/f64) ---\n"
            (format "CONDS(gold-link): raster edges=%d  gold edges=%d  lambda-sum raster=%.4f gold=%.4f\n"
                    (alength ^ints (:parent ct-g)) (alength g-ct-parent)
                    (reduce + 0.0 (filter #(not (Double/isInfinite %)) (seq ^doubles (:lambda ct-g))))
                    (reduce + 0.0 (filter #(not (Double/isInfinite %)) (seq g-ct-lambda))))
            (format "LEAVES(gold-link): raster=%d  gold=%d  set-eq=%s\n"
                    (count leaves-g) (count g-leaves) (= (set leaves-g) (set (seq g-leaves))))
            (format "BASE(gold-link) : ARI(raster, gold)=%.4f  (1.0 => condense logic exact)\n"
                    (ari labels-g (vec (seq g-base)))))]
  (spit "/tmp/diff_spine_out.txt" out)
  (println out))
