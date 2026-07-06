"""SGLang-style radix tree KV cache.

A radix tree of token-ID spans. Each node holds the materialized
:class:`DynamicKVCache` for the prefix from root to that node. Compared to
the hash-keyed :class:`PrefixCache`:

* Branching prefixes share KV automatically — e.g. one prefilled system
  prompt is reused across N divergent assistant continuations (RL rollouts,
  agent conversations).
* The longest stored prefix is returned even if no entry matches the full
  query, so partial reuse works without recomputing the matched portion.
* LRU eviction is at leaf granularity, so a hot system prompt anchored at
  an internal node never gets evicted while its leaves churn.

This implementation favours readability over absolute throughput: tree
operations are O(prefix_len) Python; the actual KV tensors are stored once
per node and cloned on read so each request gets a mutable copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx

from baby_whale_v4.cache import DynamicKVCache
from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.typing import ConfigHash, TokenizerHash


def _clone_cache(src: DynamicKVCache) -> DynamicKVCache:
    return src.clone()


@dataclass
class RadixNode:
    """One edge in the radix tree.

    ``tokens`` is this edge's span of token IDs (empty for the root).
    ``n_prefix`` is the cumulative prefix length at this node (root has 0;
    a node attached to root with edge ``[a, b, c]`` has ``n_prefix=3``).
    ``kv`` and ``last_logits`` are materialized only if this exact prefix
    has been prefilled and stored — they may be ``None`` on internal nodes
    created by a mid-edge split.
    """

    tokens: tuple[int, ...]
    n_prefix: int
    children: dict[int, RadixNode] = field(default_factory=dict)
    kv: DynamicKVCache | None = None
    last_logits: mx.array | None = None
    last_use: int = 0

    @property
    def has_payload(self) -> bool:
        return self.kv is not None and self.last_logits is not None


class RadixKVCache:
    """Token-trie KV cache keyed by ``(config_hash, tokenizer_hash)``.

    ``capacity_nodes`` bounds the total number of nodes (including internal
    split-points). When exceeded, leaves with the oldest ``last_use`` are
    evicted first; internal nodes are evicted lazily when their last child
    is removed.
    """

    def __init__(
        self,
        *,
        config: BabyWhaleV4Config,
        tokenizer_hash: TokenizerHash,
        capacity_nodes: int = 256,
    ) -> None:
        if capacity_nodes <= 0:
            raise ValueError("capacity_nodes must be positive")
        self._config_hash: ConfigHash = config.config_hash()
        self._tokenizer_hash: TokenizerHash = tokenizer_hash
        self.capacity_nodes = capacity_nodes
        self.root = RadixNode(tokens=(), n_prefix=0)
        self._clock = 0
        self.hits = 0
        self.misses = 0

    # ---- public API -------------------------------------------------------

    def match(self, prefix_ids: Sequence[int]) -> tuple[int, DynamicKVCache, mx.array] | None:
        """Return ``(n_matched, cloned_kv, last_logits)`` for the deepest stored prefix.

        Walks the tree consuming ``prefix_ids``. If no stored prefix matches
        even one token, returns ``None``. The returned KV is a fresh clone
        so the caller can mutate it without affecting the cache.
        """
        node, n = self._walk(prefix_ids)
        # If the walk landed on a non-payload node, climb up to the nearest
        # ancestor that does have payload — partial reuse beats nothing.
        while node is not None and not node.has_payload:
            if node is self.root:
                node = None
                break
            node = self._parent_of(node)
            if node is not None:
                n = node.n_prefix
        if node is None or not node.has_payload:
            self.misses += 1
            return None
        assert node.kv is not None and node.last_logits is not None
        self._clock += 1
        node.last_use = self._clock
        self.hits += 1
        return n, _clone_cache(node.kv), mx.array(node.last_logits)

    def insert(
        self,
        prefix_ids: Sequence[int],
        cache: DynamicKVCache,
        n_tokens: int,
        last_logits: mx.array,
    ) -> None:
        """Store the KV for ``prefix_ids[:n_tokens]`` at a node in the trie.

        If a prefix diverges mid-edge, the existing edge is split so the
        shared portion becomes an internal node.
        """
        if n_tokens <= 0 or n_tokens > len(prefix_ids):
            raise ValueError("n_tokens must be in (0, len(prefix_ids)]")
        target_tokens = tuple(prefix_ids[:n_tokens])
        node = self._descend_creating(target_tokens)
        self._clock += 1
        node.kv = _clone_cache(cache)
        node.last_logits = mx.array(last_logits)
        node.last_use = self._clock
        self._evict_if_needed()

    def clear(self) -> None:
        self.root = RadixNode(tokens=(), n_prefix=0)
        self._clock = 0
        self.hits = 0
        self.misses = 0

    # Stats: total node count, useful for tests + introspection.
    @property
    def n_nodes(self) -> int:
        return self._count_nodes(self.root)

    # ---- tree mechanics ---------------------------------------------------

    def _walk(self, prefix_ids: Sequence[int]) -> tuple[RadixNode | None, int]:
        """Walk the trie consuming ``prefix_ids`` as far as edges match.

        Returns ``(deepest_node, matched_count)``. The deepest_node is the
        last node we fully traversed into; if no first-token edge from the
        root matches, returns ``(root, 0)``.
        """
        node = self.root
        i = 0
        while i < len(prefix_ids):
            child = node.children.get(prefix_ids[i])
            if child is None:
                return node, i
            edge = child.tokens
            # How much of this edge matches?
            matched = 0
            while (
                matched < len(edge)
                and i + matched < len(prefix_ids)
                and edge[matched] == prefix_ids[i + matched]
            ):
                matched += 1
            if matched < len(edge):
                # Partial-match on this edge: we can't descend further.
                return node, i + matched
            # Full edge consumed; descend.
            node = child
            i += matched
        return node, i

    def _descend_creating(self, target_tokens: tuple[int, ...]) -> RadixNode:
        """Walk + split as needed, creating nodes so ``target_tokens`` ends at a node."""
        node = self.root
        i = 0
        while i < len(target_tokens):
            child = node.children.get(target_tokens[i])
            if child is None:
                # No edge: create a new child carrying the remaining span.
                new = RadixNode(
                    tokens=target_tokens[i:],
                    n_prefix=len(target_tokens),
                )
                node.children[target_tokens[i]] = new
                return new
            edge = child.tokens
            matched = 0
            while (
                matched < len(edge)
                and i + matched < len(target_tokens)
                and edge[matched] == target_tokens[i + matched]
            ):
                matched += 1
            if matched == len(edge):
                # Full edge matches: descend.
                node = child
                i += matched
                continue
            # Partial edge match: split.
            shared = edge[:matched]
            child_tail = edge[matched:]
            split = RadixNode(
                tokens=shared,
                n_prefix=node.n_prefix + matched,
            )
            # Re-parent the existing child as a descendant of `split` with
            # its tail tokens.
            child.tokens = child_tail
            split.children[child_tail[0]] = child
            node.children[shared[0]] = split
            # Now create the new branch from `split` for the remaining
            # target tokens (or return `split` if target ended exactly at
            # the split point).
            i += matched
            if i == len(target_tokens):
                return split
            new = RadixNode(
                tokens=target_tokens[i:],
                n_prefix=len(target_tokens),
            )
            split.children[target_tokens[i]] = new
            return new
        return node

    def _parent_of(self, target: RadixNode) -> RadixNode | None:
        # Linear search over the tree — fine at our capacity (256 nodes).
        # If this becomes a hot path, store parent pointers on each node.
        if target is self.root:
            return None
        stack = [self.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child is target:
                    return node
                stack.append(child)
        return None

    def _count_nodes(self, node: RadixNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children.values())

    def _evict_if_needed(self) -> None:
        while self.n_nodes > self.capacity_nodes:
            victim = self._lru_leaf()
            if victim is None or victim is self.root:
                # Nothing safe to evict (shouldn't happen with capacity > 0).
                return
            parent = self._parent_of(victim)
            if parent is None:
                return
            # Drop the victim from its parent's children dict.
            del parent.children[victim.tokens[0]]
            # If parent is now empty and has no payload, collapse it upward —
            # it was probably an internal split node that's lost its reason
            # for existing.
            if not parent.children and not parent.has_payload and parent is not self.root:
                self._evict_if_needed()  # recurse — next iteration handles parent

    def _lru_leaf(self) -> RadixNode | None:
        best: RadixNode | None = None
        best_use = -1
        stack = [self.root]
        while stack:
            node = stack.pop()
            if (
                node is not self.root
                and not node.children
                and node.has_payload
                and (best is None or node.last_use < best_use)
            ):
                best = node
                best_use = node.last_use
            stack.extend(node.children.values())
        return best
