(ns evoc.clustering-test
  "Self-contained tests for the EVoC-family clustering core (no Python dep).
  Behavioral validation against EVoC lives in dev/umap_port/*_gold.py."
  (:require [clojure.test :refer [deftest testing is]]
            [evoc.mst :as mst]
            [evoc.tree :as tree]
            [evoc.boruvka :as bor]))

(defn- rand-data [n dim seed]
  (let [r (java.util.Random. seed) a (double-array (* n dim))]
    (dotimes [i (* n dim)] (aset a i (.nextGaussian r)))
    a))

(defn- cluster [^doubles pts n dim min-samples min-cluster-size]
  (let [r (mst/mutual-reachability-mst pts n dim min-samples)
        s (mst/sort-edges r)
        ct (tree/condense (tree/linkage-tree (:from s) (:to s) (:w s) n) n min-cluster-size)]
    (vec (tree/cluster-labels ct (tree/extract-leaves ct) n))))

(deftest two-well-separated-blobs
  (testing "mutual-reachability MST + condensation recovers two obvious clusters"
    (let [pts (double-array [0.0 0.0  0.1 0.1  0.2 0.0  0.0 0.2  0.1 0.0  0.15 0.15
                             10.0 10.0  10.1 10.1  10.2 10.0  10.0 10.2  10.1 10.0  10.15 10.15])
          labels (cluster pts 12 2 2 3)]
      (is (= 2 (count (distinct (filter #(>= % 0) labels)))) "two clusters")
      (is (every? #(>= % 0) labels) "no noise on clean blobs")
      (is (apply = (take 6 labels)) "first blob shares a label")
      (is (apply = (drop 6 labels)) "second blob shares a label")
      (is (not= (first labels) (last labels)) "the two blobs get distinct labels"))))

(deftest linkage-tree-merges-in-weight-order
  (testing "single-linkage dendrogram: merged nodes get ids >= n, sizes accumulate"
    ;; 4 points, MST edges (0-1),(2-3),(1-2) ascending weight -> two pairs then join
    (let [lk (tree/linkage-tree (int-array [0 2 1]) (int-array [1 3 2])
                                (double-array [0.1 0.2 0.3]) 4)]
      ;; row0 merges {0,1} -> node 4 (size 2); row1 merges {2,3} -> node 5 (size 2);
      ;; row2 merges components 4 and 5 -> size 4
      (is (= [2 2 4] (vec (:size lk))))
      (is (= 0.3 (aget ^doubles (:delta lk) 2)))
      ;; final merge joins the two size-2 clusters (ids >= 4)
      (is (>= (aget ^ints (:left lk) 2) 4))
      (is (>= (aget ^ints (:right lk) 2) 4)))))

(deftest boruvka-mst-equals-prim
  (testing "KDTree Boruvka MST has identical total weight to dense Prim (valid min tree)"
    (let [n 250 dim 2 ms 5 X (rand-data n dim 2)
          p (mst/mutual-reachability-mst X n dim ms)
          b (bor/boruvka-mst X n dim ms)
          tot (fn [m] (reduce + 0.0 (map #(aget ^doubles (:w m) %) (range (alength ^doubles (:w m))))))]
      (is (= (clojure.core/dec n) (alength ^doubles (:w b))) "n-1 edges")
      (is (< (Math/abs (- (tot p) (tot b))) 1e-6) "same total weight as Prim"))))
