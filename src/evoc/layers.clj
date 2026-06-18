(ns evoc.layers
  "Multi-resolution cluster-layer selection — the defining EVoC feature
  (clustering.build_cluster_layers + clustering_utilities.py). Over a condensed
  tree: min_cluster_size barcode -> total persistence over size thresholds ->
  local-maxima peaks -> diverse-peak selection (Jaccard) -> per-peak cluster
  layers. fit picks labels_ = argmax(persistence)."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == quot])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == quot]]
            [evoc.tree :as tree]
            [evoc.mst :as mst]
            [clojure.set]))

;; ---- min_cluster_size_barcode (over the masked cluster tree, child>=n_samples) ----
;; cluster_tree rows come in sibling PAIRS (consecutive). Sweep in reverse by 2.
(deftm min-cluster-size-barcode!
  [ct-parent :- (Array int) ct-child :- (Array int) ct-lambda :- (Array double) ct-size :- (Array int)
   n-rows :- Long n-points :- Long min-size :- Long
   size-births :- (Array double) size-deaths :- (Array double)
   parents :- (Array int) lambda-deaths :- (Array double)] :- Long
  (dotimes [i (alength size-births)] (aset size-births i (double min-size)))
  (aset lambda-deaths 0 0.0) (aset size-deaths 0 (double n-points)) (aset parents 0 (int n-points))
  (loop [idx (- n-rows 1)]
    (when (> idx 0)
      (let [out-idx (- (long (aget ct-child idx)) n-points)
            par (long (aget ct-parent idx))
            ld (Math/exp (/ -1.0 (aget ct-lambda idx)))
            a (aget ct-size (- idx 1)) b (aget ct-size idx)
            ds (double (if (< a b) a b))]
        (aset parents (- out-idx 1) (int par)) (aset parents out-idx (int par))
        (aset lambda-deaths (- out-idx 1) ld) (aset lambda-deaths out-idx ld)
        (aset size-deaths (- out-idx 1) ds) (aset size-deaths out-idx ds)
        (let [pob (- par n-points)
              b1 (aget size-births (- out-idx 1)) b2 (aget size-births out-idx)
              mx (if (> b1 b2) b1 b2) mx (if (> ds mx) ds mx)]
          (aset size-births pob mx)))
      (recur (- idx 2))))
  n-rows)

(defn min-cluster-size-barcode
  "Returns {:births :deaths :parents :lambda-deaths} (size n_nodes)."
  [{:keys [^ints parent ^ints child ^doubles lambda ^ints size]} n-points min-size]
  (let [n-rows (alength parent)
        n-nodes (clojure.core/inc (clojure.core/- (clojure.core/aget child (clojure.core/dec n-rows))
                                                  (long n-points)))
        births (double-array n-nodes) deaths (double-array n-nodes)
        parents (int-array n-nodes) lambda-deaths (double-array n-nodes)]
    (min-cluster-size-barcode! parent child lambda size (long n-rows) (long n-points) (long min-size)
                               births deaths parents lambda-deaths)
    {:births births :deaths deaths :parents parents :lambda-deaths lambda-deaths}))

;; ---- compute_total_persistence over size thresholds (sizes = unique births) ----
(deftm total-persistence!
  [births :- (Array double) deaths :- (Array double) lambda-deaths :- (Array double) n :- Long
   sizes :- (Array double) ns :- Long tp :- (Array double)] :- (Array double)
  (dotimes [i ns] (aset tp i 0.0))
  (loop [i 1]
    (when (< i n)
      (let [birth (aget births i) death (aget deaths i) ld (aget lambda-deaths i)]
        (when (> death birth)
          (let [bidx (loop [j 0] (if (< j ns) (if (>= (aget sizes j) birth) j (recur (inc j))) ns))
                didx (loop [j 0] (if (< j ns) (if (>= (aget sizes j) death) j (recur (inc j))) ns))]
            (loop [k bidx] (when (< k didx)
                             (aset tp k (+ (aget tp k) (* (- death birth) ld)))
                             (recur (inc k)))))))
      (recur (+ i 1))))
  tp)

(defn compute-total-persistence
  "Returns {:sizes (unique births, ascending) :total-persistence}."
  [{:keys [^doubles births ^doubles deaths ^doubles lambda-deaths]}]
  (let [sizes (double-array (sort (distinct (seq births))))
        tp (double-array (alength sizes))]
    (total-persistence! births deaths lambda-deaths (alength births) sizes (alength sizes) tp)
    {:sizes sizes :total-persistence tp}))

;; ---- find_peaks: local maxima with flat-plateau handling (scipy.signal) ----
;; deftm kernel: write peak indices into `out`, return the count. (raster quot has a
;; [Long Long] overload, so the plateau midpoint stays devirtualized — bare / on two
;; longs would fall through to the boxed Number/Number Ratio path.)
(deftm find-peaks!
  [x :- (Array double) nx :- Long out :- (Array int)] :- Long
  (loop [i 1 m 0]
    (if (< i (- nx 1))
      (if (< (aget x (- i 1)) (aget x i))
        (let [ahead (loop [a (+ i 1)]
                      (if (if (< a (- nx 1)) (== (aget x a) (aget x i)) false)
                        (recur (+ a 1)) a))]
          (if (< (aget x ahead) (aget x i))
            (do (aset out m (int (quot (+ i (- ahead 1)) 2))) (recur (+ ahead 1) (+ m 1)))
            (recur (+ i 1) m)))
        (recur (+ i 1) m))
      m)))

(defn find-peaks [^doubles x]
  (let [n (alength x) out (int-array (clojure.core/quot n 2))
        m (find-peaks! x (long n) out)]
    (java.util.Arrays/copyOf out (int m))))

;; ---- select_diverse_peaks: greedy by persistence, Jaccard of active clusters ----
(defn- active-clusters [^doubles births ^doubles deaths birth]
  (set (filter (fn [i] (clojure.core/and (clojure.core/<= (clojure.core/aget births i) birth)
                                         (clojure.core/> (clojure.core/aget deaths i) birth)))
               (range (alength births)))))

(defn- jaccard [a b]
  (let [inter (count (clojure.set/intersection a b)) uni (count (clojure.set/union a b))]
    (if (clojure.core/zero? uni) 0.0 (clojure.core// (double inter) uni))))

(defn select-diverse-peaks [^ints peaks ^doubles total-persistence ^doubles sizes
                            ^doubles births ^doubles deaths min-sim max-layers]
  (if (clojure.core/zero? (alength peaks))
    (int-array 0)
    (let [sorted (sort-by (fn [p] (clojure.core/- (clojure.core/aget total-persistence p))) (seq peaks))]
      (loop [ps sorted selected [] sel-births []]
        (if (clojure.core/or (empty? ps) (clojure.core/>= (count selected) max-layers))
          (int-array selected)
          (let [p (first ps) bsize (clojure.core/aget sizes p)
                a (active-clusters births deaths bsize)
                diverse? (every? (fn [sb] (clojure.core/<= (jaccard a (active-clusters births deaths sb)) min-sim))
                                 sel-births)]
            (if diverse?
              (recur (rest ps) (conj selected p) (conj sel-births bsize))
              (recur (rest ps) selected sel-births))))))))

;; ---- extract_clusters_by_id: labels + strengths for a set of cluster node ids ----
(defn extract-clusters-by-id [condensed ^ints selected-ids n-samples]
  (let [labels (tree/cluster-label-vector condensed selected-ids 0.0 n-samples)
        strengths (tree/point-membership-strengths condensed selected-ids labels n-samples)]
    {:labels labels :strengths strengths}))

;; ---- cluster_layers from a condensed tree (the part validated vs gold) ----
(defn cluster-layers-from-condensed
  "build_cluster_layers given a condensed tree (post-spine). Returns
  {:layers [int[]] :strengths [double[]] :persistence [double] + diagnostics}."
  [condensed n-samples base-min-cluster-size min-similarity max-layers]
  (let [leaves (tree/extract-leaves condensed)
        base-labels (tree/cluster-label-vector condensed leaves 0.0 n-samples)
        base-strengths (tree/point-membership-strengths condensed leaves base-labels n-samples)
        cluster-tree (tree/mask-condensed-tree condensed n-samples)
        ^ints ct-child (:child cluster-tree)
        valid? (clojure.core/and (clojure.core/pos? (alength ct-child))
                                 (clojure.core/>= (clojure.core/aget ct-child (clojure.core/dec (alength ct-child)))
                                                  (long n-samples)))
        bc (when valid? (min-cluster-size-barcode cluster-tree n-samples base-min-cluster-size))
        tpr (when valid? (compute-total-persistence bc))
        peaks (if valid? (find-peaks (:total-persistence tpr)) (int-array 0))
        selected (if valid?
                   (select-diverse-peaks peaks (:total-persistence tpr) (:sizes tpr)
                                         (:births bc) (:deaths bc) min-similarity (clojure.core/dec max-layers))
                   (int-array 0))
        layers (atom [{:labels base-labels :strengths base-strengths :persistence 0.0}])]
    (doseq [peak (seq selected)]
      (let [best-birth (clojure.core/aget ^doubles (:sizes tpr) peak)
            persistence (clojure.core/aget ^doubles (:total-persistence tpr) peak)
            ^doubles births (:births bc) ^doubles deaths (:deaths bc)
            sel-ids (int-array (map (fn [i] (clojure.core/+ i (long n-samples)))
                                    (filter (fn [i] (clojure.core/and (clojure.core/<= (clojure.core/aget births i) best-birth)
                                                                      (clojure.core/> (clojure.core/aget deaths i) best-birth)))
                                            (range (alength births)))))
            {:keys [labels strengths]} (extract-clusters-by-id condensed sel-ids n-samples)]
        (swap! layers conj {:labels labels :strengths strengths :persistence persistence})))
    ;; sort layers by n_clusters descending
    (let [sorted (sort-by (fn [l] (clojure.core/- (clojure.core/inc (reduce clojure.core/max -1 (seq ^ints (:labels l)))))) @layers)]
      {:layers (mapv :labels sorted) :strengths (mapv :strengths sorted)
       :persistence (mapv :persistence sorted)
       :barcode bc :tp tpr :peaks peaks :selected selected})))

;; ---- full build_cluster_layers from an embedding (for the orchestrator, #39) ----
(defn build-cluster-layers
  [^doubles emb n dim {:keys [min-samples base-min-cluster-size min-similarity max-layers]
                       :or {min-samples 5 base-min-cluster-size 5 min-similarity 0.2 max-layers 10}}]
  (let [r (mst/mutual-reachability-mst emb n dim min-samples)
        s (mst/sort-edges r)
        lk (tree/linkage-tree (:from s) (:to s) (:w s) n)
        ct (tree/condense lk n base-min-cluster-size)]
    (cluster-layers-from-condensed ct n base-min-cluster-size min-similarity max-layers)))
