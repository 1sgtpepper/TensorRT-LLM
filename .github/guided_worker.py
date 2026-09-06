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

"""Exercise real guided generation and observe scheduler-selected recomputation."""

import argparse
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, PreTrainedTokenizerFast

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.llmapi import CapacitySchedulerPolicy, KvCacheConfig, SchedulerConfig
from tensorrt_llm.sampling_params import GuidedDecodingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-tokens", type=int, choices=(128, 512), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    word = "a" * 31 + "b" * 32 + "c" * 32
    vocabulary = list("abcdx") + ["<eos>"] + [f"Z{i}" for i in range(26)]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(32))), unk_token="Z25"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<eos>", pad_token="<eos>", unk_token="Z25"
    )
    assert fast.encode(word, add_special_tokens=False) == [0] * 31 + [1] * 32 + [2] * 32
    assert fast.decode([0, 1, 2, 3]) == "abcd"
    pauses = []
    original_pause = PyExecutor._pause_requests
    original_recompute = PyExecutor._pause_recompute_request

    def pause(executor: PyExecutor, requests: Iterable[LlmRequest]) -> None:
        requests = list(requests)
        for request in requests:
            pauses.append(
                {
                    "route": "pause",
                    "request_id": request.request_id,
                    "committed": request.get_tokens(0)[request.orig_prompt_len :],
                }
            )
        original_pause(executor, requests)

    def recompute(executor: PyExecutor, request: LlmRequest) -> None:
        pauses.append(
            {
                "route": "recompute",
                "request_id": request.request_id,
                "committed": request.get_tokens(0)[request.orig_prompt_len :],
            }
        )
        original_recompute(executor, request)

    report = {"kv_tokens": args.kv_tokens, "expected": word, "pauses": pauses, "phase": "setup"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    PyExecutor._pause_requests = pause
    PyExecutor._pause_recompute_request = recompute
    try:
        with tempfile.TemporaryDirectory(prefix="guided-worker-model-") as model:
            config = LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=128,
                vocab_size=32,
                max_position_embeddings=128,
                bos_token_id=4,
                eos_token_id=5,
                pad_token_id=5,
                dtype="bfloat16",
            )
            config.save_pretrained(model)
            fast.save_pretrained(model)
            with LLM(
                model,
                backend="pytorch",
                load_format="dummy",
                attn_backend="FLASHINFER",
                dtype="bfloat16",
                max_batch_size=2,
                max_seq_len=128,
                max_num_tokens=128,
                enable_chunked_prefill=False,
                cuda_graph_config=None,
                disable_overlap_scheduler=True,
                enable_autotuner=False,
                guided_decoding_backend="xgrammar",
                kv_cache_config=KvCacheConfig(
                    max_tokens=args.kv_tokens,
                    tokens_per_block=32,
                    enable_block_reuse=False,
                    free_gpu_memory_fraction=0.1,
                ),
                scheduler_config=SchedulerConfig(
                    capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
                ),
            ) as llm:
                report["phase"] = "generate"
                outputs = llm.generate(
                    [[4], [4]],
                    SamplingParams(
                        max_tokens=len(word),
                        temperature=0,
                        top_k=1,
                        end_id=5,
                        pad_id=5,
                        guided_decoding=GuidedDecodingParams(regex=word),
                    ),
                    use_tqdm=False,
                )
                tokens = [output.outputs[0].token_ids for output in outputs]
                report["tokens"] = tokens
                report["actual"] = ["".join(vocabulary[token] for token in row) for row in tokens]
                report["phase"] = "completed"
        print(json.dumps(report), flush=True)
        if args.kv_tokens == 128:
            assert any(item["committed"] for item in pauses), (
                "No post-output recomputation observed"
            )
        else:
            assert pauses == [], "The ample-capacity control unexpectedly paused"
        assert report["actual"] == [word, word], "Real worker lost guided output progress"
    finally:
        PyExecutor._pause_requests = original_pause
        PyExecutor._pause_recompute_request = original_recompute
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
