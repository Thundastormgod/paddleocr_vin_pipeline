# Session State — headline shipped, logbook audited (2026-08-21)

**STATUS: PRODUCTION CANDIDATE REGISTERED.** Suite at HEAD `8093348`:
**499 passed, 2 skipped, 7 deselected (e2e)**.

## Session 17 (2026-08-20) — Apple-GPU stack, 90% exact on test

HEAD `8093348`. Rosetta-ResNet34-IN1K (torch 2.13.0/MPS, ImageNet warm
start, 21,319,174 params): **val-102 98.73% char / 85.3% exact (87/102);
test-40 99.26% char / 90.0% exact (36/40)**, Wilson CI [0.77, 0.96].
First custom model to beat the stock engine (2.4x its test exact).
30 epochs in 51.49 min on MPS (16.7x paddle-CPU). Registered
`vin-rosetta-resnet34-torch` v1, alias `production-candidate`, weights =
ep22 `best_char_accuracy.pt`. Training run `3e340663...`; registration
completed by `register/Rosetta-ResNet34-IN1K` (`367c1fd1...`).

## Session 18 (2026-08-21) — logbook audit + run-status hygiene

LOGBOOK headline entry audited against `mlflow.db` + checkpoints; every
figure verified. One error corrected: stock val exact-match crossing is
**epoch 11** (27.45% > 25.5%), not 12 (that was the val-vs-test-figure
crossing). Entry enriched: recipe block, full run IDs, LoggedModel
`m-833ab5cc...` + Dataset linkage, ep22/ep20/ep30 selection evidence
(+3.9pp exact vs last epoch), checksum-valid rates (postproc = gate
only: 87.3→94.1 val), 51.49-min measured duration, 7x7-stem vs vd-stem
distinction. Training run status corrected FAILED→FINISHED via server
API (training completed; only the pt2 export step crashed and the
register run delivered it) — original end_time preserved, transition in
`status_history` tag. MLflow server: `mlflow ui --backend-store-uri
sqlite:///mlflow.db --port 5001` (running, PID 92304).

## Next moves

1. Torch inference integration into `VINOCRPipeline` (deployed pipeline
   is paddle-based) — required before the candidate becomes DEFAULT.
2. Larger held-out set for the 95%-target claim (n=40 CI too wide).
3. Evidence-ranked paddle mirror: Rosetta-ResNet34vd + PaddleClas
   ImageNet `ResNet34_vd_pretrained` warm start, same recipe (needs
   backbone-weight loading in the paddle trainer; ~3.75h CPU). The
   measured lever is initialization, not architecture.
4. Postprocessor in deployment: checksum GATING only, never correction.

Full details: Lumena chunks 58 (session 17), 59-60 (session 18).
