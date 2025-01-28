import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

def reward_fn_length(prompts, completions):
    """Reward function that gives higher scores to shorter completions."""
    return [-0.01 * float(len(completion)) for completion in completions]

def reward_fn_correct(prompts, completions):
    """Reward function that gives higher scores to correct completions."""
    # Look for an answer tag in the completion
    answer_tags = [re.search(r"<answer>(.*?)</answer>", completion['content']) for completion in completions]
    answers = [tag.group(1) if tag is not None else None for tag in answer_tags ]
    
    # Check if the answer is correct
    real_qs = [re.search(r"<math>(.*?)</math>", prompt['content']) for prompt in prompts]
    qs = [tag.group(1) if tag is not None else None for tag in real_qs]
    real_answers = [eval(q) if q is not None else None for q in qs]
    return [1.0 if (answer == real_answers[i] and answer is not None) else 0.0 for i, answer in enumerate(answers)]

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):

        split_queries = [query.split("<｜end▁of▁sentence｜>")[0] for query in queries]

        for i in range(len(split_queries)):
            logger.info(f"queries[{i}]: {split_queries[i]}")
        scores = [-0.01 * float(len(query)) for query in split_queries]
        logger.info(f"scores[0]: {scores}")

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")