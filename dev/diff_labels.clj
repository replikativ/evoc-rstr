(ns diff-labels
  "Isolate #37: feed raster the GOLD condensed tree + leaves, compute cluster labels
  + membership strengths, compare to the reference base_clusters / base_strengths."
  (:require [evoc.tree :as tree]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def G "/tmp/evoc_gold")
(defn ari [a b]
  (let [n (count a) pairs (fn [x] (/ (* x (dec x)) 2.0))
        cont (reduce (fn [m i] (update m [(nth a i)(nth b i)] (fnil inc 0))) {} (range n))
        ai (reduce (fn [m i] (update m (nth a i) (fnil inc 0))) {} (range n))
        bi (reduce (fn [m i] (update m (nth b i) (fnil inc 0))) {} (range n))
        sij (reduce + (map (fn [[_ v]] (pairs v)) cont))
        sa (reduce + (map (fn [[_ v]] (pairs v)) ai)) sb (reduce + (map (fn [[_ v]] (pairs v)) bi))
        exp (/ (* sa sb) (pairs n)) mx (/ (+ sa sb) 2.0)]
    (if (== mx exp) 1.0 (/ (- sij exp) (- mx exp)))))

(let [ct {:parent (npy/read-i32 (str G "/ct_parent.npy"))
          :child  (npy/read-i32 (str G "/ct_child.npy"))
          :lambda (npy/read-f64 (str G "/ct_lambda.npy"))
          :size   (npy/read-i32 (str G "/ct_child_size.npy"))}
      leaves (npy/read-i32 (str G "/leaves.npy"))
      g-base (npy/read-i32 (str G "/base_clusters.npy"))
      g-str  (npy/read-f64 (str G "/base_strengths.npy"))
      n (alength g-base)
      labels (tree/cluster-label-vector ct leaves 0.0 n)
      strengths (tree/point-membership-strengths ct leaves labels n)
      lv (vec (seq labels)) gv (vec (seq g-base))
      exact (count (filter true? (map = lv gv)))
      maxsd (reduce max 0.0 (map (fn [a b] (Math/abs (- a b))) (seq strengths) (seq g-str)))
      out (str (format "n=%d leaves=%d\n" n (alength leaves))
               (format "LABELS    : raster nclust=%d gold nclust=%d  exact-match=%d/%d  ARI=%.4f\n"
                       (count (distinct (filter #(>= % 0) lv))) (count (distinct (filter #(>= % 0) gv)))
                       exact n (ari lv gv))
               (format "STRENGTHS : max|diff|=%.3e  (raster sum=%.3f gold sum=%.3f)\n"
                       maxsd (reduce + 0.0 (seq strengths)) (reduce + 0.0 (seq g-str))))]
  (spit "/tmp/diff_labels_out.txt" out) (println out))
