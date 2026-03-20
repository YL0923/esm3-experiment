# runner.py
# Use ESM3 model to generate structure and sequence
#
# Experimental group: first generate structure, then generate sequence
# Control group (no_coords): directly generate sequence, then predict structure

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

from config import (
    MODEL_NAME, DEVICE,
    STRUCT_NUM_STEPS, STRUCT_TEMPERATURE,
    SEQ_NUM_STEPS, SEQ_TEMPERATURE,
)


def load_model(device: str = DEVICE) -> ESM3InferenceClient:
    print(f"\n[Model] Loading {MODEL_NAME} to {device} ...")
    model = ESM3.from_pretrained(MODEL_NAME).to(device)
    print("[Model] Load complete")
    return model


def run_one_sample(model: ESM3InferenceClient,
                   prompt: ESMProtein,
                   sample_id: int) -> dict | None:
    """
    Generate a single sample.

    If prompt has coordinates (experimental group):
        Step 1: generate structure first (coordinate constraints dominate)
        Step 2: then generate sequence

    If prompt has no coordinates (control group):
        Step 1: directly generate sequence (sequence-only constraint)
        Step 2: then predict structure
    """
    try:
        has_coords = (prompt.coordinates is not None)

        if has_coords:
            # Experimental group: structure first, then sequence
            protein = model.generate(
                prompt,
                GenerationConfig(
                    track="structure",
                    num_steps=STRUCT_NUM_STEPS,
                    temperature=STRUCT_TEMPERATURE,
                )
            )
            protein = model.generate(
                protein,
                GenerationConfig(
                    track="sequence",
                    num_steps=SEQ_NUM_STEPS,
                    temperature=SEQ_TEMPERATURE,
                )
            )
        else:
            # Control group: sequence first, then structure
            protein = model.generate(
                prompt,
                GenerationConfig(
                    track="sequence",
                    num_steps=SEQ_NUM_STEPS,
                    temperature=SEQ_TEMPERATURE,
                )
            )
            protein = model.generate(
                protein,
                GenerationConfig(
                    track="structure",
                    num_steps=STRUCT_NUM_STEPS,
                    temperature=STRUCT_TEMPERATURE,
                )
            )

        return {
            "sequence": protein.sequence,
            "protein":  protein,
        }

    except Exception as e:
        print(f"  [Sample {sample_id}] Generation failed: {e}")
        return None