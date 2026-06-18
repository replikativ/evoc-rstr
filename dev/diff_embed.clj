(ns diff-embed
  "Validate evoc.embed/node-embedding-layout-repr! against the reference
  node_embedding_epoch_repr. Fed identical inputs (gold init, csr, eps/epn, per-epoch
  rng_val + node_order, gamma): 1 epoch must match tightly (deterministic kernel);
  50 epochs is chaos-bounded (f32 ref vs f64 raster)."
  (:require [evoc.embed :as embed]))
(load-file (str (System/getProperty "user.home") "/Development/evoc-rstr/dev/npy.clj"))

(def G "/tmp/evoc_gold")
(defn maxabsdiff [^doubles a ^doubles b]
  (areduce a i m 0.0 (max m (Math/abs (- (aget a i) (aget b i))))))
(defn rmsnorm [^doubles a] (Math/sqrt (/ (areduce a i s 0.0 (+ s (* (aget a i) (aget a i)))) (alength a))))

(let [meta (npy/read-i32 (str G "/emb_meta.npy"))   ; [n_epochs dim V block_size]
      n-epochs (aget meta 0) dim (aget meta 1) V (aget meta 2) block (aget meta 3)
      init (npy/read-f64 (str G "/init_embedding.npy"))
      indptr (npy/read-i32 (str G "/csr_indptr.npy"))
      indices (npy/read-i32 (str G "/csr_indices.npy"))
      eps (npy/read-f64 (str G "/emb_eps.npy"))
      epn (npy/read-f64 (str G "/emb_epn.npy"))
      rng (longs (:data (npy/load-npy (str G "/emb_rng_val.npy"))))
      node-order (npy/read-i32 (str G "/emb_node_order.npy"))
      gamma (npy/read-f64 (str G "/emb_gamma.npy"))
      g-after1 (npy/read-f64 (str G "/embedding_repr_after1.npy"))
      g-after2 (npy/read-f64 (str G "/embedding_repr_after2.npy"))
      g-after5 (npy/read-f64 (str G "/embedding_repr_after5.npy"))
      g-after50 (npy/read-f64 (str G "/embedding_repr.npy"))
      run (fn [neps]
            (let [emb (double-array (alength init))
                  _ (System/arraycopy init 0 emb 0 (alength init))
                  eons (aclone ^doubles eps) eonsn (aclone ^doubles epn)
                  upd (double-array (* V dim))]
              (embed/node-embedding-layout-repr! emb indptr indices eps epn eons eonsn
                                                 rng node-order gamma upd
                                                 0.5 0.1 (long V) (long dim) (long neps) (long block))
              emb))
      e1 (run 1) e2 (run 2) e5 (run 5) e50 (run n-epochs)
      out (str (format "n_epochs=%d dim=%d V=%d block=%d\n" n-epochs dim V block)
               (format "after-1  : max|diff|=%.3e  (epoch0 no-op; raster & gold both = init: %s)\n"
                       (maxabsdiff e1 g-after1) (and (zero? (maxabsdiff e1 init)) (zero? (maxabsdiff g-after1 init))))
               (format "after-2  : max|diff|=%.3e   gold-rms=%.4f   (1 real epoch; tight => kernel correct)\n"
                       (maxabsdiff e2 g-after2) (rmsnorm g-after2))
               (format "after-5  : max|diff|=%.3e   gold-rms=%.4f   (f32/f64 drift growing)\n"
                       (maxabsdiff e5 g-after5) (rmsnorm g-after5))
               (format "after-%d : max|diff|=%.3e   gold-rms=%.4f  raster-rms=%.4f  (chaos-bounded)\n"
                       n-epochs (maxabsdiff e50 g-after50) (rmsnorm g-after50) (rmsnorm e50)))]
  (spit "/tmp/diff_embed_out.txt" out) (println out))
