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

"""Observe guided output from a real small target and draft model."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, PreTrainedTokenizerFast

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSpecMetadata, Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.eagle3_dynamic_tree import Eagle3OneModelDynamicTreeWorker
from tensorrt_llm.llmapi import Eagle3DecodingConfig, KvCacheConfig
from tensorrt_llm.sampling_params import GuidedDecodingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("linear", "tree"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    vocabulary = list("xabcdef") + ["<eos>"] + [f"Z{i}" for i in range(24)]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(32))), unk_token="Z23"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<eos>", pad_token="<eos>", unk_token="Z23"
    )
    words = ("aabcf", "aadef")
    assert fast.encode(words[0], add_special_tokens=False) == [1, 1, 2, 3, 6]
    assert fast.decode([1, 1, 4, 5, 6]) == words[1]
    trace = []
    report = {"mode": args.mode, "words": words, "trace": trace, "phase": "setup"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    worker_type = Eagle3OneModelDynamicTreeWorker if args.mode == "tree" else Eagle3OneModelWorker
    original_sample = worker_type.sample_and_accept_draft_tokens

    def sample(
        worker: Eagle3OneModelWorker,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Eagle3OneModelSpecMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guide = worker.guided_decoder
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_seqs - num_contexts
        record = {"contexts": num_contexts, "generations": num_generations}
        if args.mode == "tree" and num_generations and worker.spec_tree_manager is not None:
            storage = worker.spec_tree_manager.slot_storage
            slots = storage.all_ids_buf[num_contexts : attn_metadata.num_seqs]
            record["tree_valid"] = storage.has_tree[slots].cpu().tolist()
            record["topology"] = (
                storage.pack_retrieve_from_slots(slots, num_generations).cpu().tolist()
            )
            record["drafts"] = spec_metadata.draft_tokens.cpu().tolist()
        accepted, counts = original_sample(worker, input_ids, logits, attn_metadata, spec_metadata)
        if guide is not None and guide.requests_hostfunc is not None:
            record["requests"] = [
                {
                    "id": request.request_id,
                    "slot": request.seq_slot,
                    "new_token": request.new_token,
                    "previous_accepted_drafts": request.num_accepted_draft_tokens,
                    "drafts": request.draft_tokens,
                }
                for request in guide.requests_hostfunc.valid_requests()
            ]
            record["mask_enabled"] = guide.token_mask_host[: logits.shape[0]].tolist()
            record["advanced"] = list(guide.num_advanced_tokens)
            record["predicted"] = logits.argmax(dim=-1).cpu().tolist()
            record["counts"] = counts.cpu().tolist()
            record["accepted"] = accepted.cpu().tolist()
            trace.append(record)
        return accepted, counts

    worker_type.sample_and_accept_draft_tokens = sample
    try:
        with tempfile.TemporaryDirectory(prefix="guided-tree-model-") as directory:
            target = Path(directory) / "target"
            draft = Path(directory) / "draft"
            config = LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=128,
                vocab_size=32,
                max_position_embeddings=256,
                bos_token_id=0,
                eos_token_id=7,
                pad_token_id=7,
                dtype="bfloat16",
            )
            config.save_pretrained(target)
            config.draft_vocab_size = 32
            config.save_pretrained(draft)
            fast.save_pretrained(target)
            spec = Eagle3DecodingConfig(
                speculative_model=str(draft),
                max_draft_len=2,
                eagle3_layers_to_capture={0},
                use_dynamic_tree=args.mode == "tree",
                dynamic_tree_max_topK=32 if args.mode == "tree" else None,
                max_total_draft_tokens=64 if args.mode == "tree" else 2,
                use_rejection_sampling=False,
            )
            with LLM(
                str(target),
                backend="pytorch",
                load_format="dummy",
                attn_backend="TRTLLM",
                dtype="bfloat16",
                max_batch_size=1,
                max_seq_len=256,
                max_num_tokens=256,
                enable_chunked_prefill=False,
                cuda_graph_config=None,
                disable_overlap_scheduler=True,
                enable_autotuner=False,
                guided_decoding_backend="xgrammar",
                speculative_config=spec,
                kv_cache_config=KvCacheConfig(
                    max_tokens=512, tokens_per_block=32, enable_block_reuse=False
                ),
            ) as llm:
                report["phase"] = "generate"
                output = llm.generate(
                    [0],
                    SamplingParams(
                        max_tokens=6,
                        temperature=0,
                        top_k=1,
                        end_id=7,
                        pad_id=7,
                        guided_decoding=GuidedDecodingParams(regex="aa(bc|de)f"),
                    ),
                    use_tqdm=False,
                )
                tokens = output.outputs[0].token_ids
                report["tokens"] = tokens
                report["actual"] = fast.decode(tokens, skip_special_tokens=True)
                report["phase"] = "completed"
        print(json.dumps(report), flush=True)
        assert trace, "No guided worker execution was observed"
        if args.mode == "tree":
            assert any(any(row.get("tree_valid", [])) for row in trace), (
                "No real generated tree was consumed"
            )
        assert report["actual"] in words, "Real worker output violates the finite language"
    finally:
        worker_type.sample_and_accept_draft_tokens = original_sample
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
