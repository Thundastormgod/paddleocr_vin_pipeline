# Session State — stage-3b live under the deep-dive fixes (2026-08-19)

**STATUS: STAGE-3b TRAINING RUNNING.** HEAD `0b02a7e`. Suite: **412
passed, 11 skipped**.

## Session 14 (2026-08-19) — head-to-head verdict, C7/C8, DVC

HEAD `2daa0b6`; suite 471 passed / 2 skipped / 7 deselected (e2e); ~32 commits ahead of origin (push prepared, awaiting go).

**The measurement that reframes the project** (run `15525c80...`, in LOGBOOK):
stock PP-OCRv3 + pipeline vs from-scratch v3 on the same splits —
val-102: 82.70% char / 25.5% exact vs 66.61% / 0%; test-40: 87.79% / 37.5%
vs 68.53% / 0%. From-scratch training is refuted; production default is the
stock engine + repo pre/post-processing; every future fine-tune warm-starts
from pretrained weights and must beat the stock baseline.

Also shipped: C7 interrupt tagging + zero-epoch-resume NameError fix
(db4d899), executable training smoke `scripts/train_smoke_test.py` + weekly
CI paddle-smoke job, DVC pointer for the staged dataset (d57be14).

**Next moves:** (1) re-evaluate shelved `vin_decode` ON STOCK-ENGINE outputs —
test CER 0.122 ≈ 2 errors/plate, exactly its measured viability regime;
(2) pretrained warm-start fine-tune vs stock baseline; (3) push on user go.
