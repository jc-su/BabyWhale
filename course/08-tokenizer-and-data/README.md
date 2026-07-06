# 08 · Tokenizer & data

**Prereqs:** none · **Unlocks:** [09 · Pre-training](../09-pretraining/).

## 1 · The wall

Models eat integer ids, not text. Feed raw bytes and sequences get very long (one
step per byte). Feed whole words and your vocabulary explodes and can't spell new
words. And a *naive* BPE encoder — rescan the whole string for every merge — literally
**hangs** on long lines.

## 2 · The idea

**Byte-level BPE**: start from 256 bytes, repeatedly merge the most frequent adjacent
pair into a new token. You get a compact vocabulary that can still encode anything
(it falls back to bytes). The subtlety is *encoding speed*: this repo uses a
**heap + linked-list** encoder that applies merges in rank order in O(n·log n) instead
of O(n·merges). Then **packing** concatenates documents into fixed-length training
windows so no compute is wasted on padding.

## 🧩 From theory to code

BPE encoding as an algorithm → code:

| The idea | The code (`data/tokenizer.py`) | Why this |
|----------|--------------------------------|----------|
| repeatedly merge the highest-priority adjacent pair | `_bpe_encode` main loop | apply the learned merges in rank order |
| find the next merge in O(log n) | a min-heap keyed by merge rank | don't rescan the whole string each merge |
| splice a merged pair in O(1) | a doubly-linked list of tokens | no list reallocation per merge |

Why the heap + linked list? the naive "rescan for the best pair, rebuild the list" loop is
O(len × merges) and *hangs* on long lines; this is O(len · log len).

## 3 · In the code

- `baby_whale_v4/data/tokenizer.py` — `class ByteBPETokenizer`; `_bpe_encode` is the
  heap-based encoder (contrast the slow reference in `tests/test_bpe_tokenizer.py`).
- `baby_whale_v4/data/packing.py` — concatenate + window into training tensors.

## 4 · The payoff, measured

```bash
uv run python -m unittest tests.test_bpe_tokenizer
```

It proves the fast encoder is **output-identical** to the slow reference and encodes a
16k-character line in well under a second — the difference between "packs 15M tokens in
~52s" and "hangs forever".

## 5 · Break it & reflect

- **Reflect (🔬 systems):** tokens/char sets sequence length, which sets attention cost.
  A bigger vocab means fewer tokens but a larger embedding table — where's the crossover?

- Encode a long repeated string with a tiny vs large `merges` table — watch tokens/char.
- Train a tokenizer: `uv run baby-whale-v4 train-tokenizer --help`.

**Next:** [09 · Pre-training](../09-pretraining/) — teach the model to read.
