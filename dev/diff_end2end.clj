(ns diff-end2end
  "End-to-end EVoC gate (EVoC's own test methodology): determinism (run twice ->
  identical) + ARI-vs-truth >= EVoC thresholds + structural sanity, on 512-d blobs."
  (:require [evoc :as evoc]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def G "/tmp/evoc_gold")
(defn- tally [ks]            ; frequency map without clojure.core/update (C2-crashes on Valhalla)
  (persistent! (reduce (fn [m k] (assoc! m k (inc (get m k 0)))) (transient {}) ks)))
(defn ari [a b]
  (let [n (count a) pairs (fn [x] (/ (* x (dec x)) 2.0))
        cont (tally (map vector a b))
        ai (tally a) bi (tally b)
        sij (reduce + (map pairs (vals cont)))
        sa (reduce + (map pairs (vals ai))) sb (reduce + (map pairs (vals bi)))
        exp (/ (* sa sb) (pairs n)) mx (/ (+ sa sb) 2.0)]
    (if (== mx exp) 1.0 (/ (- sij exp) (- mx exp)))))

(let [Xm (npy/load-npy (str G "/X.npy")) [n dim] (:shape Xm)
      X (npy/read-f64 (str G "/X.npy"))
      y (vec (seq (npy/read-i32 (str G "/y_true.npy"))))
      t0 (System/nanoTime)
      r1 (evoc/fit-predict X (long n) (long dim) :seed 42)
      secs (/ (- (System/nanoTime) t0) 1.0e9)
      r2 (evoc/fit-predict X (long n) (long dim) :seed 42)
      l1 (vec (seq ^ints (:labels r1))) l2 (vec (seq ^ints (:labels r2)))
      out (str (format "n=%d dim=%d  fit=%.1fs\n" n dim secs)
               (format "DETERMINISM : labels run-twice identical = %s\n" (= l1 l2))
               (format "QUALITY     : ARI-vs-truth = %.3f  (EVoC bar >0.2; EVoC itself ~0.41)\n" (ari l1 y))
               (format "STRUCTURE   : len=%d (==n %s)  n_clusters=%d  noise=%d  layers=%d  persistence=%s\n"
                       (count l1) (= (count l1) n) (:n-clusters r1) (count (filter neg? l1))
                       (count (:layers r1)) (mapv #(format "%.1f" %) (:persistence r1))))]
  (spit "/tmp/diff_end2end_out.txt" out) (println out))
