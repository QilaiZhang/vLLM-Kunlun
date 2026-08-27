# SPDX-License-Identifier: Apache-2.0
"""Fix hybrid Mamba prefix-cache negotiation when MTP/EAGLE is enabled.

vLLM 0.25 uses the EAGLE bit for two independent purposes:

* cache the lookahead state block required by speculative decoding; and
* search one extra attention block, then drop it from the cache hit.

Mamba needs the first behavior in align mode, but not the second: draft models
do not contain Mamba layers and ``MambaManager`` cannot pop its sole state
block. Keep Mamba managers enrolled in EAGLE cache writes, while excluding
Mamba only from the extra-block lookup. Also let full attention peek beyond
``max_cache_hit_length``; the request's block-hash list is the real bound and
the EAGLE pop brings the result back to the legal hit boundary.

Off switch: ``VLLM_KUNLUN_MAMBA_EAGLE_GROUP_FIX=0``.
"""

import os

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

logger = init_logger(__name__)

_orig_find_longest_cache_hit = HybridKVCacheCoordinator.find_longest_cache_hit


def _enabled() -> bool:
    return os.getenv("VLLM_KUNLUN_MAMBA_EAGLE_GROUP_FIX", "1") == "1"


def _patched_find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    """Negotiate a hybrid hit without applying EAGLE's pop to Mamba."""
    needs_mamba_mtp_fix = any(
        isinstance(spec_group.spec, MambaSpec) and spec_group.use_eagle
        for spec_group in self.attention_groups
    )
    if not _enabled() or not needs_mamba_mtp_fix:
        return _orig_find_longest_cache_hit(
            self, block_hashes, max_cache_hit_length
        )

    def _get_block_hashes(kv_cache_spec):
        if kv_cache_spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, kv_cache_spec.block_size
        )

    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    longest_hit_length = 0
    hit_blocks_by_group = [None] * num_groups

    is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
        self.attention_groups[0].spec, FullAttentionSpec
    )
    eagle_verified: set[int] = set()

    while True:
        curr_hit_length = hit_length

        for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
            self.attention_groups
        ):
            cached_blocks = hit_blocks_by_group[group_ids[0]]
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                curr_hit_length = (
                    curr_hit_length // spec.block_size * spec.block_size
                )
                continue

            drop_eagle_block = use_eagle and idx not in eagle_verified
            lookup_length = curr_hit_length
            if drop_eagle_block and not isinstance(spec, MambaSpec):
                # max_cache_hit_length already excludes the token needed for
                # logits. The EAGLE pop removes this peeked block and lands on
                # the valid boundary.
                lookup_length = curr_hit_length + spec.block_size

            hit_blocks = manager_cls.find_longest_cache_hit(
                block_hashes=_get_block_hashes(spec),
                max_length=lookup_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=drop_eagle_block,
                alignment_tokens=self.scheduler_block_size,
            )
            new_hit_length = len(hit_blocks[0]) * spec.block_size
            if drop_eagle_block:
                eagle_verified.add(idx)
            elif new_hit_length < curr_hit_length:
                eagle_verified.clear()

            # A later cache type can constrain the common hit, never extend it.
            curr_hit_length = min(curr_hit_length, new_hit_length)
            for group_id, blocks in zip(group_ids, hit_blocks):
                hit_blocks_by_group[group_id] = blocks
            longest_hit_length = max(longest_hit_length, curr_hit_length)

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break

    first_group = self.attention_groups[0]
    if isinstance(first_group.spec, FullAttentionSpec):
        num_blocks = hit_length // first_group.spec.block_size
        for group_id in first_group.group_ids:
            if (blocks := hit_blocks_by_group[group_id]) is not None:
                del blocks[num_blocks:]

    self.num_uncached_common_prefix_tokens = longest_hit_length - hit_length
    return tuple(
        blocks if blocks is not None else [] for blocks in hit_blocks_by_group
    ), hit_length


HybridKVCacheCoordinator.find_longest_cache_hit = _patched_find_longest_cache_hit
if _enabled():
    logger.info(
        "[KunlunPlugin] enabled MTP-aware hybrid Mamba prefix-cache negotiation"
    )
