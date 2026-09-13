# Fixed same-weight FP64 validation

The selected validation completed once. All 153,580 saved inputs produced identical final float32 outputs on Torch CPU64, Torch GPU64 and ORT CPU64. Maximum new inter-backend pre-clamp difference is 0 rad, below the unchanged 1e-5 rad limit. All 601 calls and 153,580 rows per backend were attempted, returned, synchronized and verified. Manual export required zero traced forwards; no training, BFM or native physics ran.

The new ONNX SHA256 is `147a710ac8d6fd6de592c93f3ca14af4f7fcf156b970bbc3ab7d5d86f3586501`. It uses the exact ordinary 55,000 stored float32 weights and normalization, promoted to float64 execution with final float32 output. It is a separate numerical implementation. The original FP32 fit/export failure remains preserved and failed.

All 27 full new-versus-old backend difference arrays were saved. The largest drift from an old FP32 output is 1.233527820687641e-5 rad (velocity corpus versus old ORT32). Six corpus/backend pairs changed a native clipping mask; at most four components/four rows changed in a pair. These are recorded consequences of the numerical implementation, not a claim of identical behavior to the old FP32 graph.

New saved-data objectives are nominal 0.000448774919884, velocity response 0.000068037114398 and physical response 0.000095646217262. The continuation's worsening velocity-response evidence remains relevant. Numerical agreement does not qualify connected simulation stability, timing or hardware.

The durable process exited 0 with all final pins exact; both process IDs are absent. The owner receipt is `owner_completion_verification.json`, SHA256 `1036e4de6a7b1a017c217d2e361a9b9e50095fc792a2dc81c7f94338447da445`. Independent graph/output review and a separately gated witness/canonical evaluation remain required.
