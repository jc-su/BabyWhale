# EMNLP demo video — 2:30, shot-by-shot

Speaking rate ~2.4 words/sec. Each scene lists: SCREEN (what to show),
DO (your actions), and VO (say this verbatim; word counts fit the slot).

Record 1080p+, terminal font >= 18pt, browser zoom 125%, light theme.
One take per scene is fine; assemble in any editor (even QuickTime trim).

---

## Scene 0 · Title (0:00–0:08)
SCREEN: title slide — "BabyWhale: Machine-Checked Courseware for the Full
LLM Lifecycle" + github.com/jc-su/BabyWhale.
VO (18 words):
"BabyWhale teaches the full LLM lifecycle on one codebase — and its teaching
material is checked by machines."

## Scene 1 · The lifecycle, live (0:08–0:35)
SCREEN: terminal, repo root.
DO: run
    uv run python course/00-the-map/journey.py
(pre-warm once off-camera so this takes ~10 s; point cursor at the loss
numbers, then the sample).
VO (55 words):
"One command: a model is built, trained, and sampled, live on this Mac.
Watch the loss fall from six to two-point-five. The course then takes this
same journey through pre-training, mid-training, S-F-T, D-P-O, reinforcement
learning with verifiable rewards, quantization, and a continuous-batching
server — twenty-two modules, one model's biography."

## Scene 2 · The course, on the web (0:35–1:00)
SCREEN: browser — https://jc-su.github.io/BabyWhale → open
"03 · MLA". Scroll slowly: the wall → the theory-to-code table (pause 3 s)
→ the pasted source snippet.
VO (52 words):
"Every module reads the same way. The wall: why the KV cache is the memory
bottleneck. The idea, from the DeepSeek papers. Then a theory-to-code table:
every term of the math mapped to the exact line in the implementation. And
the code you see here is the real source — not a copy."

## Scene 3 · Build it, graded by the real system (1:00–1:35)
SCREEN: split — editor with course/03-attention-mla/lab_mla.py (left),
terminal (right).
DO:
  1. Run: uv run python course/03-attention-mla/lab_mla.py   -> NotImplementedError
  2. In the editor, replace the raise with the two lines:
         latent = kv @ w_down
         reconstructed = latent @ w_up
  3. Run again -> "PASS — you implemented the core of MLA."
VO (62 words):
"Now build it yourself. The exercise fails until you implement it. The core
of multi-head latent attention is two matrix products: compress to a latent,
reconstruct on the way out. Run it again — PASS. The grader compared my code
against the maintained implementation, behavior to behavior. Twenty-three
exercises work like this, and the project's own test suite is the autograder."

## Scene 4 · The docs cannot lie (1:35–1:55)
SCREEN: terminal.
DO:
    sed -i '' 's/kv_a_proj/kv_a_proj_RENAMED/g' baby_whale_v4/attention.py
    uv run python -m unittest tests.test_course_snippets     # FAILS loudly
    git checkout -- baby_whale_v4/attention.py
(zoom on the failure message: "`kv_a_proj` shown but not in attention.py (renamed?)")
VO (43 words):
"What keeps the documentation honest? Rename one function in the library and
the drift guard fails, naming the exact symbol the course still displays.
Continuous integration blocks the change until the courseware is updated.
Documentation becomes falsifiable, the same way code is."

## Scene 5 · Every claim is a command (1:55–2:15)
SCREEN: terminal.
DO: run
    uv run python course/03-attention-mla/ablation.py
(point at the 8x line), then run
    uv run python paper/experiments/verify_claims.py    # 12 x PASS scrolls by
VO (43 words):
"Every quantitative claim is a command. This module's payoff: M-L-A caches
eight times less than multi-head attention, at M-Q-A size. Even the paper is
checked — this script re-derives every number in the manuscript from the
repository. Twelve checks, all passing."

## Scene 6 · Close (2:15–2:30)
SCREEN: closing slide — lifecycle figure from the paper +
"MIT · github.com/jc-su/BabyWhale · course: jc-su.github.io/BabyWhale ·
334 tests green in public CI".
VO (26 words):
"BabyWhale is open source and runs on the Mac you already own. Clone it,
break it, and watch the tests catch you. Thanks for watching."

---

## Pre-recording checklist
- [ ] `uv run python -m unittest discover -s tests` green on this machine
- [ ] Run journey.py once off-camera (Metal warmup) so Scene 1 is fast
- [ ] Open browser tab on Module 03 in advance (Scene 2 starts scrolled to top)
- [ ] Put the two solution lines in a scratch buffer to paste in Scene 3
- [ ] After Scene 4, CONFIRM the revert: `git status` must be clean
- [ ] Mic check; speak ~5% slower than feels natural
- [ ] If any scene runs long, cut Scene 5's verify_claims run first
