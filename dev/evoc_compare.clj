(ns evoc-compare
  "Run evoc-rstr on the demo dataset (same X as the reference) and dump labels for
  the side-by-side plot. Run from a classpath with raster + umap-rstr + evoc-rstr."
  (:require [evoc :as evoc]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def D "/tmp/evoc_demo")
(let [m (npy/load-npy (str D "/X.npy")) [n dim] (:shape m)
      X (npy/read-f64 (str D "/X.npy"))
      t0 (System/nanoTime)
      res (evoc/fit-predict X (long n) (long dim) :seed 42)
      secs (/ (- (System/nanoTime) t0) 1.0e9)
      labels ^ints (:labels res)
      ;; save labels as a (n,1) f64 npy (python rounds to int)
      out (double-array n)
      _ (dotimes [i n] (aset out i (double (aget labels i))))]
  (npy/save-f64-2d (str D "/rstr_labels.npy") out n 1)
  (spit (str D "/rstr_summary.txt")
        (format "evoc-rstr: %.1fs  clusters=%d noise=%d  layers=%d"
                secs (:n-clusters res) (count (filter neg? (seq labels))) (count (:layers res))))
  (println (slurp (str D "/rstr_summary.txt"))))
