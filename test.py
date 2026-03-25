import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import GenerationConfig

from config import (
    MODEL_NAME, DEVICE,
    STRUCT_NUM_STEPS, STRUCT_TEMPERATURE,
    SEQ_NUM_STEPS, SEQ_TEMPERATURE,
    BASE_SEED,
    CONDITIONS,
)
from structure import load_pdb, build_prompt


def main():
    print("=" * 60)
    print("DEBUG: Path2 structure drift check")
    print("=" * 60)

    # ---- load WT ----
    wt = load_pdb()

    # ---- choose one condition（你可以换）
    condition = CONDITIONS[-1]  # mask最多的那个最明显
    mask_layers = condition["mask_layers"]

    print(f"\nUsing condition: {condition['label']}")
    print(f"Mask layers: {mask_layers}")

    # ---- build prompt ----
    prompt = build_prompt(wt, mask_layers)

    # ---- load model ----
    print("\n[Model] Loading...")
    model = ESM3.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("[Model] Ready")

    seed = BASE_SEED
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # Step 1: sequence generation
    # ============================================================
    print("\n[Step 1] Generating sequence...")

    with torch.inference_mode():
        p2 = model.generate(
            prompt,
            GenerationConfig(
                track="sequence",
                num_steps=SEQ_NUM_STEPS,
                temperature=SEQ_TEMPERATURE,
            )
        )

    coords_before = p2.coordinates.clone()

    # ============================================================
    # Step 2: structure generation
    # ============================================================
    print("\n[Step 2] Generating structure...")

    with torch.inference_mode():
        p2 = model.generate(
            p2,
            GenerationConfig(
                track="structure",
                num_steps=STRUCT_NUM_STEPS,
                temperature=STRUCT_TEMPERATURE,
            )
        )

    coords_after = p2.coordinates

    # ============================================================
    # Compare
    # ============================================================
    print("\n[Compare] Checking coordinate drift...")

    total = 0
    changed = 0
    changed_positions = []

    L = coords_before.shape[0]

    for i in range(L):
        before = coords_before[i]
        after = coords_after[i]

        # 只看“原本不是 NaN 的位置”
        if torch.isnan(before).any():
            continue

        total += 1

        # 判断是否变化（阈值可调）
        diff = torch.norm(before - after, dim=-1).mean()

        if diff > 1e-3:
            changed += 1
            changed_positions.append((i + 1, diff.item()))

    print(f"\nTotal non-NaN positions: {total}")
    print(f"Changed positions:       {changed}")
    print(f"Change ratio:            {changed / total:.3f}")

    # 打印前20个变化最大的
    changed_positions.sort(key=lambda x: -x[1])

    print("\nTop changed residues:")
    for pos, diff in changed_positions[:20]:
        print(f"  Residue {pos:3d} | mean shift = {diff:.4f} Å")

    print("\n[Conclusion hint]")
    if changed / total > 0.3:
        print("⚠️ Large portion of backbone changed → global refolding happened")
    else:
        print("✅ Mostly local changes only")


if __name__ == "__main__":
    main()