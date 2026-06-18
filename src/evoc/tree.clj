(ns evoc.tree
  "HDBSCAN*-family condensation (ports EVoC's cluster_trees.py):
  sorted MST -> mst_to_linkage_tree -> condense_tree -> extract_leaves ->
  get_cluster_label_vector. Tree/index bookkeeping; numeric loops are deftm,
  thin defn glue for allocation/assembly."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= ==])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= ==]]))

;; --- linkage union-find (EVoC linkage_merge_*): merged nodes get ids >= n ---
;; Find root with path compression (linkage_merge_find).
(deftm lk-find! [parent :- (Array int) node0 :- Long] :- Long
  (let [node (loop [node node0]
               (let [p (long (aget parent node))]
                 (if (and (not (== p -1)) (not (== p node)))
                   (recur p)
                   node)))]
    (aset parent node (int node))
    (loop [relabel node0]
      (if (not (== (long (aget parent relabel)) node))
        (let [nxt (long (aget parent relabel))]
          (aset parent relabel (int node))
          (recur nxt))
        node))))

;; mst_to_linkage_tree: each sorted MST edge merges two components into a new
;; dendrogram node (id >= n). Output rows: [bigger_child, smaller_child, delta, size].
(deftm linkage-tree!
  [smst-from :- (Array int) smst-to :- (Array int) smst-delta :- (Array double) n :- Long
   lk-left :- (Array int) lk-right :- (Array int) lk-delta :- (Array double) lk-size :- (Array int)]
  :- (Array int)
  (let [m (alength smst-delta)
        sz (* 2 n)
        parent (int-array sz)
        node-size (int-array sz)]
    (dotimes [i sz] (aset parent i (int -1)))
    (dotimes [i n] (aset node-size i 1))
    (loop [index 0 nxt n]
      (if (< index m)
        (let [lc (lk-find! parent (long (aget smst-from index)))
              rc (lk-find! parent (long (aget smst-to index)))
              big (if (> lc rc) lc rc)
              small (if (> lc rc) rc lc)
              csz (+ (aget node-size lc) (aget node-size rc))]
          (aset lk-left index (int big))
          (aset lk-right index (int small))
          (aset lk-delta index (aget smst-delta index))
          (aset lk-size index (int csz))
          (aset node-size nxt (int csz))
          (aset parent lc (int nxt))
          (aset parent rc (int nxt))
          (recur (inc index) (inc nxt)))
        lk-left)))
  lk-left)

;; Thin glue: allocate the 4 linkage columns and run the kernel.
(defn linkage-tree
  [^ints sfrom ^ints sto ^doubles sdelta n]
  (let [m (alength sdelta)
        lk-left (int-array m) lk-right (int-array m)
        lk-delta (double-array m) lk-size (int-array m)]
    (linkage-tree! sfrom sto sdelta (long n) lk-left lk-right lk-delta lk-size)
    {:left lk-left :right lk-right :delta lk-delta :size lk-size}))

;; --- condensation (bfs_from_hierarchy / eliminate_branch / condense_tree) ---

;; FIFO BFS over the dendrogram from `root`, written into `order` (used as the
;; queue itself). Node id >= n is internal: its children are lk-left/right[node-n].
;; Returns the number of nodes visited.
(deftm bfs-from! [lk-left :- (Array int) lk-right :- (Array int) n :- Long
                  root :- Long order :- (Array int)] :- Long
  (aset order 0 (int root))
  (loop [head 0 tail 1]
    (if (< head tail)
      (let [node (long (aget order head))]
        (if (>= node n)
          (let [i (- node n)]
            (aset order tail (aget lk-left i))
            (aset order (+ tail 1) (aget lk-right i))
            (recur (+ head 1) (+ tail 2)))
          (recur (+ head 1) tail)))
      tail)))

;; Record a branch falling out of its parent cluster at `lambda`: a singleton
;; point becomes one condensed edge; an internal branch is BFS'd, its leaf points
;; recorded and its internal nodes marked `ignore`. Returns the new write index.
(deftm eliminate-branch!
  [branch :- Long parent-node :- Long lambda :- Double
   lk-left :- (Array int) lk-right :- (Array int) n :- Long
   c-parent :- (Array int) c-child :- (Array int) c-lambda :- (Array double) c-size :- (Array int)
   idx :- Long ignore :- (Array int) wq :- (Array int)] :- Long
  (if (< branch n)
    (do (aset c-parent idx (int parent-node)) (aset c-child idx (int branch))
        (aset c-lambda idx lambda) (aset c-size idx 1)
        (+ idx 1))
    (let [cnt (bfs-from! lk-left lk-right n branch wq)]
      (loop [t 0 i idx]
        (if (< t cnt)
          (let [sub (long (aget wq t))]
            (if (< sub n)
              (do (aset c-parent i (int parent-node)) (aset c-child i (int sub))
                  (aset c-lambda i lambda) (aset c-size i 1)
                  (recur (+ t 1) (+ i 1)))
              (do (aset ignore sub 1) (recur (+ t 1) i))))
          i)))))

;; condense_tree: BFS the dendrogram; at each internal node split into legitimate
;; sub-clusters (both children >= min_cluster_size) or fall points out otherwise.
;; Returns the number of condensed-tree edges written.
(deftm condense-tree!
  [lk-left :- (Array int) lk-right :- (Array int) lk-delta :- (Array double) lk-size :- (Array int)
   n :- Long min-cluster-size :- Long
   c-parent :- (Array int) c-child :- (Array int) c-lambda :- (Array double) c-size :- (Array int)
   relabel :- (Array int) ignore :- (Array int) node-list :- (Array int) wq :- (Array int)] :- Long
  (let [root (- (* 2 n) 2)
        nl-count (bfs-from! lk-left lk-right n root node-list)]
    (dotimes [i (alength c-size)] (aset c-size i 1))
    (aset relabel root (int n))
    (loop [t 0 idx 0 next-label (+ n 1)]
      (if (< t nl-count)
        (let [node (long (aget node-list t))]
          (if (or (== (aget ignore node) 1) (< node n))
            (recur (+ t 1) idx next-label)
            (let [parent-node (long (aget relabel node))
                  i (- node n)
                  left (long (aget lk-left i))
                  right (long (aget lk-right i))
                  d (aget lk-delta i)
                  lambda (if (> d 0.0) (/ 1.0 d) (/ 1.0 0.0))
                  lc (if (>= left n) (long (aget lk-size (- left n))) 1)
                  rc (if (>= right n) (long (aget lk-size (- right n))) 1)]
              (cond
                (and (< lc min-cluster-size) (>= rc min-cluster-size))
                (do (aset relabel right (int parent-node))
                    (recur (+ t 1)
                           (eliminate-branch! left parent-node lambda lk-left lk-right n
                                              c-parent c-child c-lambda c-size idx ignore wq)
                           next-label))
                (and (>= lc min-cluster-size) (< rc min-cluster-size))
                (do (aset relabel left (int parent-node))
                    (recur (+ t 1)
                           (eliminate-branch! right parent-node lambda lk-left lk-right n
                                              c-parent c-child c-lambda c-size idx ignore wq)
                           next-label))
                (and (< lc min-cluster-size) (< rc min-cluster-size))
                (let [idx2 (eliminate-branch! left parent-node lambda lk-left lk-right n
                                              c-parent c-child c-lambda c-size idx ignore wq)
                      idx3 (eliminate-branch! right parent-node lambda lk-left lk-right n
                                              c-parent c-child c-lambda c-size idx2 ignore wq)]
                  (recur (+ t 1) idx3 next-label))
                :else
                (do (aset relabel left (int next-label))
                    (aset c-parent idx (int parent-node)) (aset c-child idx (int next-label))
                    (aset c-lambda idx lambda) (aset c-size idx (int lc))
                    (aset relabel right (int (+ next-label 1)))
                    (aset c-parent (+ idx 1) (int parent-node)) (aset c-child (+ idx 1) (int (+ next-label 1)))
                    (aset c-lambda (+ idx 1) lambda) (aset c-size (+ idx 1) (int rc))
                    (recur (+ t 1) (+ idx 2) (+ next-label 2)))))))
        idx))))

(defn condense
  "Thin glue: linkage tree + min-cluster-size -> condensed tree
   {:parent :child :lambda :size} (trimmed to the written length)."
  [{:keys [^ints left ^ints right ^doubles delta ^ints size]} n min-cluster-size]
  (let [n (long n)
        cap (clojure.core/* 2 n)
        c-parent (int-array cap) c-child (int-array cap)
        c-lambda (double-array cap) c-size (int-array cap)
        relabel (int-array cap) ignore (int-array cap)
        node-list (int-array cap) wq (int-array cap)
        idx (condense-tree! left right delta size n (long min-cluster-size)
                            c-parent c-child c-lambda c-size relabel ignore node-list wq)]
    {:parent (java.util.Arrays/copyOf c-parent idx)
     :child (java.util.Arrays/copyOf c-child idx)
     :lambda (java.util.Arrays/copyOf c-lambda idx)
     :size (java.util.Arrays/copyOf c-size idx)}))

;; --- leaf extraction + labelling (extract_leaves / get_cluster_label_vector) ---

;; A cluster is a leaf iff it never has a child that is itself a cluster (size>1).
;; leaf-ind sized n-nodes: 1 for candidate cluster ids [n-points,n-nodes), cleared
;; for any parent with a size>1 child.
(deftm extract-leaves-indicator!
  [c-parent :- (Array int) c-size :- (Array int) m :- Long n-points :- Long
   leaf-ind :- (Array int)] :- (Array int)
  (dotimes [i (alength leaf-ind)] (aset leaf-ind i (int (if (< i n-points) 0 1))))
  (dotimes [e m]
    (when (> (aget c-size e) 1) (aset leaf-ind (aget c-parent e) (int 0))))
  leaf-ind)

;; union-find with path-halving + union-by-rank (EVoC disjoint_set ds_*)
(deftm ds-find! [dp :- (Array int) x0 :- Long] :- Long
  (loop [x x0]
    (let [p (long (aget dp x))]
      (if (not (== p x))
        (do (aset dp x (int (aget dp p)))      ; path-halving
            (recur p))
        x))))

(deftm ds-union! [dp :- (Array int) dr :- (Array int) a :- Long b :- Long] :- Long
  (let [x0 (ds-find! dp a) y0 (ds-find! dp b)]
    (if (== x0 y0)
      x0
      (let [swap (< (aget dr x0) (aget dr y0))
            x (if swap y0 x0)
            y (if swap x0 y0)]
        (aset dp y (int x))
        (when (== (aget dr x) (aget dr y))
          (aset dr x (int (+ (aget dr x) 1))))
        x))))

;; get_cluster_label_vector: union every condensed edge whose child is NOT a
;; selected leaf cluster, then each point's root = its leaf cluster id (or noise).
(deftm cluster-labels!
  [c-parent :- (Array int) c-child :- (Array int) m :- Long n-samples :- Long
   dp :- (Array int) dr :- (Array int) is-leaf :- (Array int) result :- (Array int)
   root-cluster :- Long label-map :- (Array int)] :- (Array int)
  (dotimes [i (alength dp)] (aset dp i (int i)) (aset dr i (int 0)))
  (dotimes [e m]
    (when (== (aget is-leaf (aget c-child e)) 0)
      (ds-union! dp dr (long (aget c-parent e)) (long (aget c-child e)))))
  (dotimes [p n-samples]
    (let [cl (ds-find! dp p)]
      (aset result p (if (<= cl root-cluster) (int -1) (aget label-map cl)))))
  result)

(defn- imin [^ints a] (areduce a i acc Long/MAX_VALUE (clojure.core/min acc (clojure.core/aget a i))))
(defn- imax [^ints a] (areduce a i acc Long/MIN_VALUE (clojure.core/max acc (clojure.core/aget a i))))

(defn extract-leaves
  "Leaf clusters of the condensed tree (ascending ids), matching extract_leaves."
  [{:keys [^ints parent ^ints size]}]
  (if (clojure.core/zero? (alength parent))
    (int-array 0)
    (let [n-points (imin parent)
          n-nodes (clojure.core/inc (imax parent))
          leaf-ind (int-array n-nodes)]
      (extract-leaves-indicator! parent size (alength parent) (long n-points) leaf-ind)
      (int-array (filter (fn [i] (clojure.core/== 1 (clojure.core/aget leaf-ind i)))
                         (range n-points n-nodes))))))

(defn cluster-labels
  "Point -> cluster label vector (noise = -1), matching get_cluster_label_vector
   (multi-cluster case)."
  [{:keys [^ints parent ^ints child]} ^ints leaves n-samples]
  (let [n-nodes (clojure.core/max (clojure.core/inc (imax parent)) (clojure.core/inc (imax child)))
        root-cluster (imin parent)
        is-leaf (int-array n-nodes)
        label-map (int-array n-nodes -1)
        _ (dotimes [r (alength leaves)]
            (let [l (clojure.core/aget leaves r)]
              (aset is-leaf l (int 1)) (aset label-map l (int r))))   ; leaves already ascending
        dp (int-array n-nodes) dr (int-array n-nodes)
        result (int-array n-samples)]
    (cluster-labels! parent child (alength parent) (long n-samples)
                     dp dr is-leaf result (long root-cluster) label-map)
    result))

;; --- single-cluster label vector (get_single_cluster_label_vector) ---
(deftm single-cluster-labels!
  [c-parent :- (Array int) c-child :- (Array int) c-lambda :- (Array double) m :- Long
   cluster :- Long epsilon :- Double n-samples :- Long result :- (Array int)] :- (Array int)
  (dotimes [i n-samples] (aset result i (int -1)))
  (let [max-lambda (loop [i 0 mx 0.0]
                     (if (< i m)
                       (recur (inc i) (if (== (aget c-parent i) cluster)
                                        (if (> (aget c-lambda i) mx) (aget c-lambda i) mx) mx))
                       mx))]
    (dotimes [i m]
      (let [n (long (aget c-child i)) cur (aget c-lambda i)]
        (when (< n n-samples)
          (if (> epsilon 0.0)
            (aset result n (int (if (>= cur (/ 1.0 epsilon)) 0 -1)))
            (when (>= cur max-lambda) (aset result n (int 0))))))))
  result)

;; --- point membership strengths (get_point_membership_strength_vector) ---
;; deaths[cluster] = max lambda over child_size==1 edges with parent==cluster (max_lambdas);
;; strength = 1 if death is 0 / edge lambda is infinite, else min(lambda, death)/death.
(deftm membership-strengths!
  [c-parent :- (Array int) c-child :- (Array int) c-lambda :- (Array double) c-size :- (Array int)
   m :- Long n-samples :- Long root-cluster :- Long
   labels :- (Array int) label-to-cluster :- (Array int) deaths :- (Array double)
   result :- (Array double)] :- (Array double)
  (dotimes [i (alength deaths)] (aset deaths i 0.0))
  (dotimes [e m]
    (when (== (aget c-size e) 1)
      (let [par (long (aget c-parent e)) lam (aget c-lambda e)]
        (when (> lam (aget deaths par)) (aset deaths par lam)))))
  (dotimes [i n-samples] (aset result i 0.0))
  (dotimes [e m]
    (let [point (long (aget c-child e))]
      (when (and (< point root-cluster) (>= (aget labels point) 0))
        (let [cluster (long (aget label-to-cluster (aget labels point)))
              death (aget deaths cluster)
              lam (aget c-lambda e)]
          (aset result point
                (if (or (== death 0.0) (== lam (/ 1.0 0.0)))
                  1.0
                  (/ (if (< lam death) lam death) death)))))))
  result)

(defn mask-condensed-tree
  "Keep only condensed-tree edges whose child is an internal cluster node
  (child >= n-samples) — EVoC's cluster_tree = mask_condensed_tree(ct, child>=n)."
  [{:keys [^ints parent ^ints child ^doubles lambda ^ints size]} n-samples]
  (let [m (alength parent)
        keep (filterv (fn [e] (clojure.core/>= (clojure.core/aget child e) (long n-samples))) (range m))
        k (count keep)
        p (int-array k) c (int-array k) l (double-array k) s (int-array k)]
    (dotimes [i k]
      (let [e (nth keep i)]
        (aset p i (clojure.core/aget parent e)) (aset c i (clojure.core/aget child e))
        (aset l i (clojure.core/aget lambda e)) (aset s i (clojure.core/aget size e))))
    {:parent p :child c :lambda l :size s}))

(defn cluster-label-vector
  "get_cluster_label_vector: dispatch single-cluster vs multi-cluster. `clusters`
  are condensed-tree cluster ids; epsilon = cluster_selection_epsilon."
  [{:keys [^ints parent ^ints child ^doubles lambda] :as ct} ^ints clusters epsilon n-samples]
  (cond
    (clojure.core/zero? (alength parent)) (int-array n-samples -1)
    (clojure.core/== 1 (alength clusters))
    (let [result (int-array n-samples)]
      (single-cluster-labels! parent child lambda (alength parent)
                              (long (clojure.core/aget clusters 0)) (double epsilon)
                              (long n-samples) result)
      result)
    :else
    (cluster-labels ct (int-array (sort (seq clusters))) n-samples)))

(defn point-membership-strengths
  "get_point_membership_strength_vector. labels = cluster-label-vector output."
  [{:keys [^ints parent ^ints child ^doubles lambda ^ints size]} ^ints clusters ^ints labels n-samples]
  (if (clojure.core/zero? (alength parent))
    (double-array n-samples)
    (let [n-nodes (clojure.core/max (clojure.core/inc (imax parent)) (clojure.core/inc (imax child)))
          root-cluster (imin parent)
          sorted-clusters (int-array (sort (seq clusters)))
          deaths (double-array n-nodes)
          result (double-array n-samples)]
      (membership-strengths! parent child lambda size (alength parent) (long n-samples)
                             (long root-cluster) labels sorted-clusters deaths result)
      result)))
