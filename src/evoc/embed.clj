(ns evoc.embed
  "EVoC node embedding — ports node_embedding_epoch. Structurally identical to
  UMAP's optimize-layout! (per-edge attractive + negative-sampled SGD over the
  fuzzy graph), but with EVoC's grad coefficients and a deterministic uint32-
  arithmetic negative-sample index (not Tausworthe):

    attractive: dist=√d², gc = (-2·noise·dist - 2)/(2·d² - 0.5·dist + 1)   [no clip]
    negative  : k = ((n+p)·i·rng) mod n_vertices (uint32 wrap); if d²>1e-2:
                gc = 4/((1+0.25·d²)·d²); grad clipped to [-4,4]

  Uses a local uclip (gradient clip to [-4,4]). The whole multi-epoch solve compiles to one method."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == mod bit-and])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == sqrt mod bit-and]]))

;; Clamp gradient into [-4, 4] (umap.layouts.clip) — local to keep clustering
;; independent of the UMAP layer (umap-rstr).
(deftm uclip [v :- Double] :- Double
  (if (> v 4.0) 4.0 (if (< v -4.0) -4.0 v)))

;; ---------------------------------------------------------------------------
;; Reproducible epoch — ports node_embedding_epoch_repr (the DEFAULT seeded EVoC
;; path; node_embedding_init=None still uses the seeded repr epoch). It:
;;   - iterates per NODE over CSR neighbours (not per edge)
;;   - move ONLY from_node (deferred), accumulate into `updates`, apply per block
;;   - deterministic neg index node_order[(raw*(ep+p+1)*rng) mod V]
;;   - gamma schedule on the repulsive term; updates decay (1-alpha)^2*0.5 / epoch
;;   - node-order is a per-epoch permutation, flat (n-epochs * V), offset ep*V
;; `updates` is caller-zeroed scratch (V*dim); it persists+decays across epochs.
(deftm node-embedding-layout-repr!
  [emb :- (Array double) csr-indptr :- (Array int) csr-indices :- (Array int)
   eps :- (Array double) epn :- (Array double) eons :- (Array double) eonsn :- (Array double)
   rng-vals :- (Array long) node-order :- (Array int) gamma-vals :- (Array double)
   updates :- (Array double)
   noise :- Double init-alpha :- Double n-vertices :- Long dim :- Long n-epochs :- Long block-size :- Long]
  :- (Array double)
  (dotimes [ep n-epochs]
    (let [epd (double ep)
          rng (aget rng-vals ep)
          gamma (aget gamma-vals ep)
          alpha (if (== ep 0)
                  init-alpha
                  (* init-alpha (- 1.0 (/ (double (dec ep)) (double n-epochs)))))
          no-base (* (long ep) n-vertices)]
      ;; process nodes in blocks; deferred updates applied per block
      (loop [bs 0]
        (when (< bs n-vertices)
          (let [be (if (< (+ bs block-size) n-vertices) (+ bs block-size) n-vertices)]
            ;; --- compute phase: accumulate into updates[from_node] ---
            (loop [ni bs]
              (when (< ni be)
                (let [from (long (aget node-order (+ no-base ni)))
                      fb (* from dim)]
                  (loop [raw (long (aget csr-indptr from))]
                    (when (< raw (long (aget csr-indptr (+ from 1))))
                      (when (<= (aget eons raw) epd)
                        (let [to (long (aget csr-indices raw))
                              tb (* to dim)
                              d2 (loop [d 0 acc 0.0]
                                   (if (< d dim)
                                     (recur (inc d) (let [df (- (aget emb (+ fb d)) (aget emb (+ tb d)))]
                                                      (+ acc (* df df)))) acc))]
                          (when (> d2 0.0)
                            (let [dist (sqrt d2)
                                  gc (/ (- (* (* -2.0 noise) dist) 2.0)
                                        (+ (- (* 2.0 d2) (* 0.5 dist)) 1.0))]
                              (dotimes [d dim]
                                (aset updates (+ fb d)
                                      (+ (aget updates (+ fb d))
                                         (* (* gc (- (aget emb (+ fb d)) (aget emb (+ tb d)))) alpha))))))
                          (aset eons raw (+ (aget eons raw) (aget eps raw)))
                          (let [n-neg (long (/ (- epd (aget eonsn raw)) (aget epn raw)))]
                            (dotimes [p n-neg]
                              (let [idx (mod (* (* raw (+ (long ep) (long p) 1)) rng) n-vertices)
                                    to2 (long (aget node-order (+ no-base idx)))
                                    tb2 (* to2 dim)
                                    d2n (loop [d 0 acc 0.0]
                                          (if (< d dim)
                                            (recur (inc d) (let [df (- (aget emb (+ fb d)) (aget emb (+ tb2 d)))]
                                                             (+ acc (* df df)))) acc))]
                                (when (> d2n 0.01)
                                  (let [gcn (/ (* gamma 4.0) (* (+ 1.0 (* 0.25 d2n)) d2n))]
                                    (when (> gcn 0.0)
                                      (dotimes [d dim]
                                        (aset updates (+ fb d)
                                              (+ (aget updates (+ fb d))
                                                 (* (uclip (* gcn (- (aget emb (+ fb d)) (aget emb (+ tb2 d))))) alpha)))))))))
                            (aset eonsn raw (+ (aget eonsn raw) (* (double n-neg) (aget epn raw)))))))
                      (recur (+ raw 1))))
                  (recur (+ ni 1)))))
            ;; --- apply phase: emb[from_node] += updates[from_node] ---
            (loop [ni bs]
              (when (< ni be)
                (let [from (long (aget node-order (+ no-base ni))) fb (* from dim)]
                  (dotimes [d dim]
                    (aset emb (+ fb d) (+ (aget emb (+ fb d)) (aget updates (+ fb d)))))
                  (recur (+ ni 1)))))
            (recur be))))
      ;; decay the deferred buffer (momentum) for the next epoch
      (let [decay (* (* (- 1.0 alpha) (- 1.0 alpha)) 0.5)]
        (dotimes [i (* n-vertices dim)] (aset updates i (* (aget updates i) decay))))))
  emb)
