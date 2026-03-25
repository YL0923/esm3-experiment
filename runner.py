# runner.py
# Call ESM3 model to generate structure and sequence
# Path 1 (structure-first): generate structure -> generate sequence

import gc
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

from config import (
    MODEL_NAME, DEVICE,
    STRUCT_NUM_STEPS, STRUCT_TEMPERATURE,
    SEQ_NUM_STEPS, SEQ_TEMPERATURE,
    BASE_SEED,
)


def load_model(device: str = DEVICE) -> ESM3InferenceClient:
    print(f"\n[Model] Loading {MODEL_NAME} on {device} ...")
    model = ESM3.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("[Model] Loaded successfully")
    return model


def _cleanup_cuda():
    """Release cached CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_one_sample(model: ESM3InferenceClient,
                   prompt: ESMProtein,
                   sample_id: int) -> dict | None:
    """
    Generate a single sample via Path 1 (structure-first):
        prompt -> generate structure -> generate sequence
    Returns None if generation fails.
    """
    seed = BASE_SEED + sample_id

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            p1 = model.generate(
                prompt,
                GenerationConfig(
                    track="structure",
                    num_steps=STRUCT_NUM_STEPS,
                    temperature=STRUCT_TEMPERATURE,
                )
            )
            p1 = model.generate(
                p1,
                GenerationConfig(
                    track="sequence",
                    num_steps=SEQ_NUM_STEPS,
                    temperature=SEQ_TEMPERATURE,
                )
            )

    except torch.cuda.OutOfMemoryError as e:
        print(f"  [Sample {sample_id}] Failed: CUDA OOM: {e}")
        _cleanup_cuda()
        return None
    except Exception as e:
        print(f"  [Sample {sample_id}] Failed: {e}")
        _cleanup_cuda()
        return None

    result = {
        "struct_sequence": p1.sequence,
        "struct_protein":  p1,
    }

    _cleanup_cuda()
    return result