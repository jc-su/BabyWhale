# EMNLP demo video — script & shot list (target: 3:30, max 5:00)

Record at 1080p+, terminal font ≥16pt, browser at 125% zoom. One continuous
cut per scene is fine; no music needed. Screen + voiceover.

## Scene 0 · Title card (0:00–0:10)
Slide: "BabyWhale — a test-verified, laptop-scale course through the full LLM
lifecycle" + repo URL. VO: one sentence of the abstract's first claim.

## Scene 1 · The whole lifecycle in 30 seconds (0:10–0:50)
Terminal:
```bash
uv run python course/00-the-map/journey.py
```
VO while it trains: "One tiny model, born, trained, and sampled — live, on
this laptop. The course makes each leg of this loop good."
Point at the printed loss falling (~6.0 → ~2.5) and the sample.

## Scene 2 · Read a module (0:50–1:40)
Browser: https://jc-su.github.io/BabyWhale/ → Module 03 (MLA).
Scroll slowly through: the wall → the theory-to-code table (pause 3s) →
the pasted real code. VO: "Every module bridges the equation to the exact
lines in the source — and that pasted code is CI-guarded against drift."

## Scene 3 · Build it (1:40–2:30)
Terminal, split with editor:
```bash
uv run python course/03-attention-mla/lab_mla.py      # NotImplementedError
# edit lab_mla.py:  latent = kv @ w_down ; reconstructed = latent @ w_up
uv run python course/03-attention-mla/lab_mla.py      # PASS
```
VO: "Twenty-three labs like this — graded by the project's own test suite
against the real implementation. You can't fake a green."

## Scene 4 · The docs cannot lie (2:30–3:00)
Terminal:
```bash
# temporarily rename kv_a_proj in baby_whale_v4/attention.py, then:
uv run python -m unittest tests.test_course_snippets    # FAILS loudly
git checkout -- baby_whale_v4/attention.py
```
VO: "Rename one function and the documentation's drift guard fails CI until
the docs are updated."

## Scene 5 · Measure it (3:00–3:20)
```bash
uv run python course/03-attention-mla/ablation.py      # the 8x table
```
VO: "Every payoff in the course is a command that prints the number."

## Scene 6 · Close (3:20–3:30)
Slide: lifecycle figure from the paper + "MIT · github.com/jc-su/BabyWhale ·
334 tests green in public CI". VO: one closing sentence.

## Pre-recording checklist
- [ ] `uv run python -m unittest discover -s tests` green on the machine
- [ ] journey.py warm-run once (Metal shader compile) so Scene 1 is fast
- [ ] Undo the Scene-4 rename before recording anything else
- [ ] Terminal theme: light background records better on projectors
