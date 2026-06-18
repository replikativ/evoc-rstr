(ns diff-layers
  "Isolate #38 multi-res on the GOLD condensed tree: compare barcode, total
  persistence, peaks, selected peaks, and final layers/persistence to the gold."
  (:require [evoc.layers :as L] [evoc.tree :as tree]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def G "/tmp/evoc_gold")
(defn maxd [a b] (reduce max 0.0 (map (fn [x y] (Math/abs (- (double x) (double y)))) a b)))

(let [ct {:parent (npy/read-i32 (str G "/ct_parent.npy")) :child (npy/read-i32 (str G "/ct_child.npy"))
          :lambda (npy/read-f64 (str G "/ct_lambda.npy")) :size (npy/read-i32 (str G "/ct_child_size.npy"))}
      params (npy/read-i32 (str G "/params.npy"))   ; [n_neighbors min_samples n_epochs base_mcs n_comp max_layers]
      base-mcs (aget params 3) max-layers (aget params 5)
      n (alength (npy/read-i32 (str G "/base_clusters.npy")))
      g-births (npy/read-f64 (str G "/bc_births.npy")) g-deaths (npy/read-f64 (str G "/bc_deaths.npy"))
      g-ld (npy/read-f64 (str G "/bc_lambda_deaths.npy")) g-tp (npy/read-f64 (str G "/total_persistence.npy"))
      g-sizes (npy/read-f64 (str G "/sizes.npy")) g-peaks (npy/read-i32 (str G "/peaks.npy"))
      g-sel (npy/read-i32 (str G "/selected_peaks.npy")) g-persist (npy/read-f64 (str G "/persistence_scores.npy"))
      ;; raster multi-res on the GOLD condensed tree
      cluster-tree (tree/mask-condensed-tree ct n)
      bc (L/min-cluster-size-barcode cluster-tree n base-mcs)
      tpr (L/compute-total-persistence bc)
      peaks (L/find-peaks (:total-persistence tpr))
      sel (L/select-diverse-peaks peaks (:total-persistence tpr) (:sizes tpr) (:births bc) (:deaths bc) 0.2 (dec max-layers))
      res (L/cluster-layers-from-condensed ct n base-mcs 0.2 max-layers)
      out (str (format "n=%d base_mcs=%d max_layers=%d  cluster_tree edges=%d\n" n base-mcs max-layers (alength ^ints (:child cluster-tree)))
               (format "BARCODE  : births max|d|=%.3e  deaths max|d|=%.3e  lambda_deaths max|d|=%.3e (n_nodes r=%d g=%d)\n"
                       (maxd (seq (:births bc)) (seq g-births)) (maxd (seq (:deaths bc)) (seq g-deaths))
                       (maxd (seq (:lambda-deaths bc)) (seq g-ld)) (alength ^doubles (:births bc)) (alength g-births))
               (format "PERSIST  : sizes match=%s  total_persistence max|d|=%.3e (r=%d g=%d)\n"
                       (= (vec (seq (:sizes tpr))) (vec (seq g-sizes))) (maxd (seq (:total-persistence tpr)) (seq g-tp))
                       (alength ^doubles (:total-persistence tpr)) (alength g-tp))
               (format "PEAKS    : raster=%s gold=%s  match=%s\n" (vec (seq peaks)) (vec (seq g-peaks)) (= (vec (seq peaks)) (vec (seq g-peaks))))
               (format "SELECTED : raster=%s gold=%s  match=%s\n" (vec (seq sel)) (vec (seq g-sel)) (= (set (seq sel)) (set (seq g-sel))))
               (format "LAYERS   : raster n_layers=%d gold n_layers=%d  persistence raster=%s gold=%s\n"
                       (count (:layers res)) (alength g-persist)
                       (mapv #(format "%.2f" %) (:persistence res)) (mapv #(format "%.2f" %) (seq g-persist))))]
  (spit "/tmp/diff_layers_out.txt" out) (println out))
