# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dynamic-tree grammar investigation for published source 17803f6c0cbe8d9fed3ccf46a295b867e3493d3e.

Run only with matching Python/native TensorRT-LLM code in the fork GPU environment.
These tests do not establish full-model reachability or capture the executor lifecycle.
Full worker/model reachability remains a separate validation requirement.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    GuidedDecoder,
    GuidedRequest,
    GuidedRequests,
)
from tensorrt_llm._torch.speculative.dynamic_tree_ops import DynamicTreeOpsConverter
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig


def _make_guided_decoder(width: int) -> tuple[GuidedDecoder, GuidedDecodingParams]:
    """Use the public raw-vocabulary configuration with the real matcher backend."""
    vocab = list("abcdefx") + ["<eos>"] + [f"Z{i}" for i in range(24)]
    config = GuidedDecodingConfig(encoded_vocab=vocab, stop_token_ids=[7])
    params = GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, "aa(bc|de)f")
    decoder = GuidedDecoder(
        config, max_num_sequences=1, vocab_size_padded=32, max_num_draft_tokens=width
    )
    context = GuidedRequest(
        guided_decoding_params=params,
        request_id=41,
        seq_slot=0,
        is_context_init_state=True,
        is_last_context_chunk=True,
        draft_tokens=[],
    )
    requests = GuidedRequests([context], 1, 0, width)
    assert decoder._build(requests) == [], "Context matcher initialization failed"
    return decoder, params


def _build_tree(
    top_k: int, selected: list[int], parents: list[int]
) -> tuple[DynamicTreeOpsConverter, torch.Tensor]:
    count = len(selected) + 1
    converter = DynamicTreeOpsConverter(top_k, 2, count - 1, 1, torch.device("cuda"))
    mask = torch.empty((1, count, 1), dtype=torch.int32, device="cuda")
    position, index, child, sibling = [
        torch.empty((1, count), dtype=torch.int32, device="cuda") for _ in range(4)
    ]
    converter.build_dynamic_tree(
        torch.tensor([parents], dtype=torch.int64, device="cuda"),
        torch.tensor([selected], dtype=torch.int64, device="cuda"),
        mask,
        position,
        index,
        child,
        sibling,
        use_packed_mask=True,
    )
    return converter, torch.stack([index, child, sibling], dim=-1)


@pytest.mark.parametrize("prefer_second_branch", [False, True])
def test_guided_tree_output_obeys_ancestor_path(prefer_second_branch: bool) -> None:
    """A finite-language oracle is independent of the production grammar matcher."""
    decoder, params = _make_guided_decoder(4)
    converter, packed = _build_tree(2, [0, 1, 2, 4], [0, 0, 1])
    assert packed.cpu().tolist() == [
        [[0, 1, -1], [1, 3, 2], [2, 4, -1], [3, -1, -1], [4, -1, -1]]
    ], "Native tree fixture does not represent the intended two branches"
    # Context emitted a. First generation has no valid tree; use an illegal
    # first draft so this step leaves no tentative grammar state to roll back.
    bootstrap = GuidedRequest(
        guided_decoding_params=params,
        request_id=41,
        seq_slot=0,
        is_generation_in_progress_state=True,
        new_token=0,
        draft_tokens=[6, 6, 6, 6],
        num_accepted_draft_tokens=0,
    )
    bootstrap_requests = GuidedRequests([bootstrap], 0, 1, 4)
    bootstrap_logits = torch.full((5, 32), -20.0, dtype=torch.float32, device="cuda")
    bootstrap_logits[0, 0] = 10
    assert decoder._build(bootstrap_requests) == [], "Bootstrap matcher build failed"
    decoder._copy_bitmask(bootstrap_requests)
    decoder._apply_bitmask(bootstrap_requests, bootstrap_logits)
    bootstrap_predicted = bootstrap_logits.argmax(dim=-1).to(torch.int32).reshape(1, 5)
    _, bootstrap_count, bootstrap_tokens = converter.verify_dynamic_tree_greedy_out_packed(
        torch.tensor([[0, 6, 6, 6, 6]], dtype=torch.int32, device="cuda"),
        packed,
        bootstrap_predicted,
        1,
        3,
        tree_valid=torch.zeros(1, dtype=torch.bool, device="cuda"),
    )
    assert int(bootstrap_count.item()) == 0, "Invalid-tree bootstrap must accept no drafts"
    assert int(bootstrap_tokens[0, 0].item()) == 0, "Bootstrap must emit the second a"

    # The following generation now has a valid tree. No rollback call is
    # inserted: the dynamic worker omits it, and the bootstrap needs none.
    request = GuidedRequest(
        guided_decoding_params=params,
        request_id=41,
        seq_slot=0,
        is_generation_in_progress_state=True,
        new_token=0,
        draft_tokens=[1, 3, 2, 4],
        num_accepted_draft_tokens=0,
    )
    requests = GuidedRequests([request], 0, 1, 4)
    logits = torch.full((5, 32), -20.0, dtype=torch.float32, device="cuda")
    logits[0, 1] = 9 if prefer_second_branch else 10
    logits[0, 3] = 10 if prefer_second_branch else 9
    logits[1, 2] = 10
    logits[2, 6] = 20  # x has high finite score; after aad, only e is legal.
    logits[2, 4] = 10
    logits[3:, 5] = 10
    original_logits = logits.cpu().clone()
    assert decoder._build(requests) == [], "Generation matcher build failed"
    decoder._copy_bitmask(requests)
    decoder._apply_bitmask(requests, logits)
    predicted = logits.argmax(dim=-1).to(torch.int32).reshape(1, 5)
    candidates = torch.tensor([[0, 1, 3, 2, 4]], dtype=torch.int32, device="cuda")
    candidates[:, 0] = predicted[:, 0]
    _, accepted_count, tokens = converter.verify_dynamic_tree_greedy_out_packed(
        candidates, packed, predicted, 1, 3
    )
    count = int(accepted_count.item()) + 1
    emitted = "aa" + "".join("abcdefx"[tid] for tid in tokens[0, :count].cpu().tolist())
    words = ("aabcf", "aadef")
    assert any(word.startswith(emitted) for word in words), (
        f"Emitted prefix {emitted!r} violates the language; "
        f"token_mask={decoder.token_mask_host[:5].tolist()}, "
        f"accepted_drafts={count - 1}"
    )
    # Derive the greedy constrained output separately from exact allowed characters.
    node_for_prefix = {"aa": 0, "aab": 1, "aad": 2, "aabc": 3, "aade": 4}
    expected = "aa"
    while expected not in words:
        allowed = [
            letter
            for letter in "abcdefx"
            if any(word.startswith(expected + letter) for word in words)
        ]
        row = node_for_prefix[expected]
        expected += max(allowed, key=lambda c: float(original_logits[row, "abcdefx".index(c)]))
    assert emitted == expected, f"Wrong constrained greedy output: {emitted!r} != {expected!r}"
