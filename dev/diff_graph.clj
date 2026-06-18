(ns diff-graph
  "Isolate the fuzzy-graph stage: feed umap.graph the GOLD kNN (inds+dists) and
  compare the symmetrized fuzzy graph to the reference gold graph COO."
  (:require [umap.graph :as graph]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def G "/tmp/evoc_gold")
(let [nn-inds (npy/read-i32 (str G "/nn_inds.npy"))
      nn-dists (npy/read-f64 (str G "/nn_dists.npy"))
      gr (npy/read-i32 (str G "/graph_row.npy"))
      gc (npy/read-i32 (str G "/graph_col.npy"))
      gd (npy/read-f64 (str G "/graph_data.npy"))
      shp (npy/load-npy (str G "/nn_inds.npy"))
      [n k] (:shape shp) n (long n) k (long k)
      sigmas (double-array n) rhos (double-array n)
      _ (graph/smooth-knn-dist! nn-dists n k sigmas rhos)
      mvals (double-array (* n k))
      _ (graph/membership-strengths! nn-inds nn-dists sigmas rhos n k mvals)
      {:keys [head tail weights]} (graph/symmetrize nn-inds mvals n k)
      ;; canonical undirected edge -> weight maps
      key (fn [i j] (+ (* (long (min i j)) n) (long (max i j))))
      rmap (persistent! (reduce (fn [m e] (assoc! m (key (aget ^ints head e) (aget ^ints tail e))
                                                  (aget ^doubles weights e)))
                                (transient {}) (range (alength ^doubles weights))))
      gmap (persistent! (reduce (fn [m e] (let [i (aget gr e) j (aget gc e)]
                                            (if (== i j) m (assoc! m (key i j) (aget gd e)))))
                                (transient {}) (range (alength gd))))
      ks (into #{} (concat (keys rmap) (keys gmap)))
      maxd (reduce (fn [mx kk] (max mx (Math/abs (- (double (get rmap kk 0.0)) (double (get gmap kk 0.0)))))) 0.0 ks)
      missing (count (filter #(or (not (contains? rmap %)) (not (contains? gmap %))) ks))
      out (str (format "n=%d k=%d\n" n k)
               (format "GRAPH : raster undirected edges=%d  gold undirected edges=%d\n" (count rmap) (count gmap))
               (format "        edges only-in-one=%d  max|w-diff|=%.6e\n" missing maxd)
               (format "        raster wsum=%.4f  gold wsum=%.4f\n"
                       (reduce + 0.0 (vals rmap)) (reduce + 0.0 (vals gmap))))]
  (spit "/tmp/diff_graph_out.txt" out) (println out))
