import json
from typing import List, Dict, Any, Optional
import torch

import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load .env and auto-login
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("CRITICAL: vLLM not installed. Run 'pip install vllm'")
    LLM = None
    SamplingParams = None

ChatMessage = Dict[str, str]

# Load your model mapping
try:
    with open("config/model_config.json", "r") as f:
        mapping_name_dict = json.load(f)
except FileNotFoundError:
    mapping_name_dict = {}

class VLLMOfflineClient:
    """
    Dedicated client for Offline Batch Inference.
    """
    def __init__(self, model_key: str, gpu_utilization: float = 0.9, options: Dict[str, Any] = None):
        if not LLM:
            raise ImportError("vLLM library is missing.")

        # 1. Resolve Model Path
        if model_key in mapping_name_dict:
            model_info = mapping_name_dict[model_key]
            if isinstance(model_info, dict):
                self.model_path = model_info.get("vllm", model_key)
            else:
                self.model_path = model_info
        else:
            self.model_path = model_key

        print(f"--- [GPU] Loading Model: {self.model_path} ---")
        
        # 2. Instantiate Engine
        # enforce_eager=False allows CUDA Graphs (Faster)
        self.llm_engine = LLM(
            model=self.model_path,
            gpu_memory_utilization=gpu_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            tensor_parallel_size=2
        )
        
        self.tokenizer = self.llm_engine.get_tokenizer()
        
        # 3. Setup Default Sampling Params
        opts = options or {}
        self.sampling_params = SamplingParams(
            temperature=opts.get("temperature", 0.7),
            max_tokens=opts.get("max_tokens", 1024),
            seed=opts.get("seed", 42)
        )

    def chat_batch(self, messages_list: List[List[ChatMessage]]) -> List[str]:
        """
        Processes a whole list of distinct conversations in parallel.
        """
        # 1. Pre-process text (Apply Chat Template)
        prompts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,enable_thinking=False  # Disable thinking for Qwen models
)
            for msgs in messages_list
        ]

        # 2. Generate (Batch)
        # use_tqdm=False here because we will use our own TQDM in main.py
        outputs = self.llm_engine.generate(prompts, self.sampling_params, use_tqdm=True)

        # 3. Extract text
        return [output.outputs[0].text for output in outputs]
