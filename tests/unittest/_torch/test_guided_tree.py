# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableTreeGuidedDecoder
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.dynamic_tree_ops import DynamicTreeOpsConverter
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig


@pytest.fixture(params=list(GuidedDecodingConfig.GuidedDecodingBackend))
def guide(request):
    vocabulary = list("xabcdef") + ["<eos>"] + [f"Z{i}" for i in range(24)]
    tokenizer = Tokenizer(models.BPE(dict(zip(vocabulary, range(32))), merges=[], unk_token="Z23"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.ByteLevel()
    config = GuidedDecodingConfig(
        backend=request.param,
        encoded_vocab=vocabulary,
        tokenizer_str=tokenizer.to_str(),
        stop_token_ids=[7],
    )
    decoder = CapturableTreeGuidedDecoder(config, 2, 32, max_num_draft_tokens=5)
    yield decoder, vocabulary
    torch.cuda.synchronize()


def _request(decoder, regex, slot=0, request_id=1):
    request = LlmRequest(
        request_id=request_id,
        seq_slot=slot,
        max_new_tokens=16,
        input_tokens=[0],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        end_id=7,
        guided_decoding_params=GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, regex),
    )
    request.py_seq_slot = slot
    batch = ScheduledRequests()
    batch.append_context_request(request)
    decoder.add_batch(batch)
    logits = torch.zeros((1, 32), device="cuda")
    decoder.execute(logits)
    token = int(logits.argmax(-1).item())
    assert token == 1
    request.add_new_token(token, 0)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.py_batch_idx = slot
    return request


def _generation(decoder, requests, new_tokens=None):
    batch = ScheduledRequests()
    for request in requests:
        batch.append_generation_request(request)
    decoder.add_batch(batch, new_tokens=new_tokens)


@pytest.mark.parametrize("tree_valid", [False, True])
@pytest.mark.parametrize(
    "words,drafts,prefixes",
    [
        (("abcf", "adef"), ["x", "b", "d", "c", "e"], ["ax", "ab", "ad", "abc", "ade"]),
        (("abac", "aabc"), ["x", "b", "a", "a", "b"], ["ax", "ab", "aa", "aba", "aab"]),
        (("a", "ab", "ad"), ["<eos>", "b", "d", "x", "x"], [None, "ab", "ad", "abx", "adx"]),
    ],
)
def test_tree_masks_follow_ancestors(guide, tree_valid, words, drafts, prefixes):
    decoder, vocabulary = guide
    request = _request(decoder, "|".join(words))
    request.py_draft_tokens = [vocabulary.index(token) for token in drafts]
    _generation(decoder, [request])
    # Root children 1, 2, 3; nodes 4 and 5 descend from 2 and 3 respectively.
    retrieve = torch.tensor(
        [[[0, 1, -1], [1, -1, 2], [2, 4, 3], [3, 5, -1], [4, -1, -1], [5, -1, -1]]],
        dtype=torch.int32,
        device="cuda",
    )
    logits = torch.zeros((6, 32), device="cuda")
    decoder.execute(logits, retrieve=retrieve, tree_valid=torch.tensor([tree_valid], device="cuda"))
    finite = logits.isfinite().cpu()
    for row, prefix in enumerate(["a", *prefixes]):
        if row and (
            not tree_valid or prefix is None or not any(w.startswith(prefix) for w in words)
        ):
            continue
        expected = {
            vocabulary.index(w[len(prefix)])
            for w in words
            if w.startswith(prefix) and len(w) > len(prefix)
        }
        if prefix in words:
            expected.add(7)
        assert set(finite[row].nonzero().flatten().tolist()) == expected
    # Speculative traversal must restore the root, including after EOS.
    matcher = decoder.grammar_matchers[request.py_seq_slot]
    assert not matcher.is_terminated()
    matcher.fill_next_token_bitmask(decoder.bitmask_host, 0)
    decoder.copy_bitmask(num_bitmask_tokens=1)
    root_logits = torch.zeros((1, 32), device="cuda")
    decoder.apply_bitmask(root_logits, num_bitmask_tokens=1)
    assert torch.equal(root_logits.isfinite().cpu()[0], finite[0])


@pytest.mark.parametrize(
    "preferred,partial,expected",
    [("b", False, "bcf"), ("d", False, "def"), ("f", False, "f"), ("b", True, "bc")],
)
def test_verified_branch_continues_with_overlap_tokens(guide, preferred, partial, expected):
    decoder, vocabulary = guide
    request = _request(decoder, "a(bc|de)?f")
    drafts = ["x", "b", "d", "x" if partial else "c", "e"]
    request.py_draft_tokens = [vocabulary.index(token) for token in drafts]
    _generation(decoder, [request])
    retrieve = torch.tensor(
        [[[0, 1, -1], [1, -1, 2], [2, 4, 3], [3, 5, -1], [4, -1, -1], [5, -1, -1]]],
        dtype=torch.int32,
        device="cuda",
    )
    valid = torch.ones(1, dtype=torch.bool, device="cuda")
    logits = torch.zeros((6, 32), device="cuda")
    logits[0, vocabulary.index(preferred)] = 10
    decoder.execute(logits, retrieve=retrieve, tree_valid=valid)
    target = logits.argmax(-1).to(torch.int32).unsqueeze(0)
    candidates = torch.tensor([[1, *request.py_draft_tokens]], dtype=torch.int32, device="cuda")
    converter = DynamicTreeOpsConverter(3, 2, 5, 1, torch.device("cuda"))
    _, draft_counts, accepted = converter.verify_dynamic_tree_greedy_out_packed(
        candidates, retrieve, target, 1, 3, tree_valid=valid
    )
    counts = draft_counts + 1
    decoder.commit_tree_tokens(accepted, counts)
    # Match production overlap: the request history is still old, while the
    # verifier's final output supplies the next root through device staging.
    next_tokens = torch.zeros((6, 2, 1), dtype=torch.int32, device="cuda")
    next_tokens[0, 0, 0] = accepted.gather(1, (counts - 1).long().unsqueeze(1))[0, 0]
    _generation(decoder, [request], new_tokens=next_tokens)
    next_logits = torch.zeros((6, 32), device="cuda")
    decoder.execute(next_logits)
    count = int(counts[0].item())
    actual = "".join(vocabulary[tid] for tid in accepted[0, :count].tolist())
    assert actual == expected
    allowed = set(next_logits[0].isfinite().nonzero().flatten().tolist())
    assert allowed == ({6} if partial else {7})
    assert request.get_tokens(0) == [0, 1]


def test_graph_replay_uses_current_tree_and_clears_padded_rows(guide):
    decoder, vocabulary = guide
    logits = torch.zeros((12, 32), device="cuda")
    retrieve = torch.tensor(
        [[[0, 1, -1], [1, -1, 2], [2, 4, 3], [3, 5, -1], [4, -1, -1], [5, -1, -1]]] * 2,
        dtype=torch.int32,
        device="cuda",
    )
    valid = torch.ones(2, dtype=torch.bool, device="cuda")
    dummy = LlmRequest(
        request_id=99,
        seq_slot=0,
        max_new_tokens=16,
        input_tokens=[0],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    dummy.state = LlmRequestState.GENERATION_IN_PROGRESS
    _generation(decoder, [dummy])
    decoder.execute(logits, retrieve=retrieve, tree_valid=valid)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits.zero_()
        decoder.execute(logits, retrieve=retrieve, tree_valid=valid)
    for iteration, slots in enumerate(((0, 1), (1,))):
        requests = [
            _request(decoder, "a(bc|de)f", slot, 10 + 2 * iteration + slot) for slot in slots
        ]
        for request in requests:
            request.py_draft_tokens = [vocabulary.index(t) for t in ["x", "b", "d", "c", "e"]]
        _generation(decoder, requests)
        # Reused slots without a built tree must not consume stale links.
        valid.fill_(iteration == 0)
        graph.replay()
        finite = logits.isfinite().cpu()
        assert set(finite[0].nonzero().flatten().tolist()) == {2, 4}
        if iteration == 0:
            assert set(finite[2].nonzero().flatten().tolist()) == {3}
            assert set(finite[3].nonzero().flatten().tolist()) == {5}
            assert set(finite[6].nonzero().flatten().tolist()) == {2, 4}
        else:
            assert finite[1:6].all()
            assert finite[6:].all()


def test_tree_masks_in_a_mixed_context_generation_batch(guide):
    decoder, vocabulary = guide
    generation = _request(decoder, "a(bc|de)f")
    generation.py_draft_tokens = [vocabulary.index(t) for t in ["x", "b", "d", "c", "e"]]
    context = LlmRequest(
        request_id=2,
        seq_slot=1,
        max_new_tokens=16,
        input_tokens=[0],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        end_id=7,
        guided_decoding_params=GuidedDecodingParams(GuidedDecodingParams.GuideType.REGEX, "df"),
    )
    context.py_seq_slot = 1
    batch = ScheduledRequests()
    batch.append_context_request(context)
    batch.append_generation_request(generation)
    decoder.add_batch(batch)
    retrieve = torch.tensor(
        [[[0, 1, -1], [1, -1, 2], [2, 4, 3], [3, 5, -1], [4, -1, -1], [5, -1, -1]]],
        dtype=torch.int32,
        device="cuda",
    )
    logits = torch.zeros((7, 32), device="cuda")
    decoder.execute(
        logits, retrieve=retrieve, tree_valid=torch.ones(1, dtype=torch.bool, device="cuda")
    )
    finite = logits.isfinite().cpu()
    for row, allowed in [(0, {4}), (1, {2, 4}), (3, {3}), (4, {5}), (5, {6}), (6, {6})]:
        assert set(finite[row].nonzero().flatten().tolist()) == allowed
