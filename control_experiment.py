# control_experiment.py
# Control experiment: fix random (non-catalytic) residue sets instead of the
# catalytic core, then mask everything else.  Compares with the main
# experiment's G5 condition to test whether catalytic-site constraints have
# special importance beyond generic local-structure anchoring.
#
# Uses the same generation pipeline (structure.py, runner.py) as the main
# experiment, only the prompt construction differs.
#
# Usage:
#   python control_experiment.py
#
# Output:
#   results/results_control.txt   — per-sample and summary results
#   pdbs_control/                 — best PDB per control group

import math
import random
from pathlib import Path

from config import (
    MODEL_NAME, DEVICE, N_SAMPLES,
    STRUCT_NUM_STEPS, STRUCT_TEMPERATURE,
    SEQ_NUM_STEPS, SEQ_TEMPERATURE,
    BASE_SEED,
)
from structure import load_pdb, build_prompt, global_align, compute_rmsd, compute_lddt_ca
from runner import load_model, run_one_sample


# ============================================================
# 1. CA II configuration (same as main experiment)
# ============================================================
CA2_CONFIG = {
    "name":       "CA2",
    "pdb_path":   "1CA2.pdb",
    "pdb_chain":  "A",
    "seq_length": 256,
    "layer_1": {61, 91, 93, 116, 195},       # catalytic core (5 residues)
    "layer_2": {4, 59, 64, 118, 139, 194, 205},
}

# All layer residues for building the "mask everything except fixed" prompt
ALL_LAYERS = {
    1: CA2_CONFIG["layer_1"],
    2: CA2_CONFIG["layer_2"],
    # layers 3-6 copied from config.py for CA2
    3: {26, 27, 56, 57, 58, 60, 62, 63, 65, 66, 87, 88, 89, 90, 92, 94, 95,
        100, 101, 102, 103, 104, 111, 112, 113, 114, 115, 117, 119, 138, 140,
        141, 142, 143, 144, 145, 153, 156, 157, 177, 180, 181, 196, 206, 207,
        208, 211, 212, 219, 222, 223, 237, 238, 239, 240, 241, 242},
    4: {3, 10, 25, 28, 53, 55, 67, 86, 96, 97, 98, 99, 105, 106, 110, 120,
        137, 146, 150, 152, 154, 155, 159, 160, 163, 164, 166, 169, 172, 175,
        176, 178, 179, 182, 192, 193, 197, 198, 199, 200, 203, 204, 209, 210,
        213, 214, 218, 220, 221, 224, 225, 226, 227, 236, 243, 244},
    5: {2, 5, 8, 9, 13, 20, 24, 29, 30, 44, 46, 47, 48, 51, 52, 54, 68, 69,
        74, 75, 76, 85, 108, 109, 121, 127, 128, 130, 131, 136, 147, 148, 149,
        151, 158, 161, 162, 165, 167, 168, 170, 171, 173, 174, 183, 184, 185,
        186, 187, 188, 189, 190, 191, 201, 202, 215, 216, 217, 228, 235, 245},
    6: {1, 6, 7, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 49, 50, 70, 71, 72, 73, 77,
        78, 79, 80, 81, 82, 83, 84, 107, 122, 123, 124, 125, 126, 129, 132,
        133, 134, 135, 229, 230, 231, 232, 233, 234, 246, 247, 248, 249, 250,
        251, 252, 253, 254, 255, 256},
}


# ============================================================
# 2. Generate random control sets
# ============================================================
def generate_random_fixed_sets(seq_length, core_size, n_sets, exclude, seed=123):
    """
    Generate n_sets of random residue sets, each of size core_size,
    excluding residues in `exclude` (the real catalytic core + shell).
    """
    rng = random.Random(seed)
    candidates = [i for i in range(1, seq_length + 1) if i not in exclude]
    sets = []
    for _ in range(n_sets):
        chosen = set(rng.sample(candidates, core_size))
        sets.append(chosen)
    return sets


# ============================================================
# 3. Build prompt: fix only the given residue set, mask all others
# ============================================================
def build_control_prompt(wt_protein, fixed_residues):
    """
    Like G5 but with an arbitrary set of fixed residues instead of Layer 1.
    Fixed residues keep their sequence and coordinates; everything else is masked.
    """
    from esm.sdk.api import ESMProtein
    import torch

    wt_seq = wt_protein.sequence
    wt_coords = wt_protein.coordinates.clone()
    L = len(wt_seq)

    seq_list = list(wt_seq)
    prompt_coords = wt_coords.clone()

    for pos in range(1, L + 1):
        if pos not in fixed_residues:
            seq_list[pos - 1] = '_'
            prompt_coords[pos - 1] = float('nan')

    prompt_seq = ''.join(seq_list)
    n_masked = prompt_seq.count('_')
    print(f"[Control Prompt] Fixed: {L - n_masked} | Masked: {n_masked} | Total: {L}")

    return ESMProtein(sequence=prompt_seq, coordinates=prompt_coords)


# ============================================================
# 4. Main
# ============================================================
def main():
    print("=" * 60)
    print("  Control Experiment: Random Fixed Residues vs Catalytic Core")
    print("=" * 60)

    # Setup
    L = CA2_CONFIG["seq_length"]
    core = CA2_CONFIG["layer_1"]
    shell = CA2_CONFIG["layer_2"]
    core_size = len(core)  # 5 residues

    # Generate 3 random sets of 5 residues (excluding core + shell)
    random_sets = generate_random_fixed_sets(
        seq_length=L,
        core_size=core_size,
        n_sets=3,
        exclude=core | shell,
        seed=123,
    )

    # Load model and WT structure
    model = load_model()
    wt_protein = load_pdb(CA2_CONFIG["pdb_path"], CA2_CONFIG["pdb_chain"])
    wt_coords = wt_protein.coordinates
    wt_sequence = wt_protein.sequence

    all_positions = set(range(1, L + 1))

    # Output
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("pdbs_control").mkdir(parents=True, exist_ok=True)
    output_path = Path("results") / "results_control.txt"

    with open(output_path, 'w', encoding='utf-8') as out:

        def write(line=""):
            print(line)
            out.write(line + '\n')

        write("=" * 60)
        write("  Control Experiment Results — CA II")
        write("=" * 60)

        # ---- Run each random control set ----
        for set_idx, fixed_set in enumerate(random_sets):
            write("")
            write("=" * 60)
            write(f"  Random Control Set {set_idx + 1}")
            write(f"  Fixed residues (1-based): {sorted(fixed_set)}")
            write(f"  Number fixed: {len(fixed_set)}")
            write("=" * 60)

            prompt = build_control_prompt(wt_protein, fixed_set)
            records = []

            for sample_id in range(N_SAMPLES):
                write(f"\n  --- Sample {sample_id + 1}/{N_SAMPLES} ---")

                result = run_one_sample(model, prompt, sample_id)

                if result is None:
                    write("  Generation failed")
                    continue

                coords = result["struct_protein"].coordinates
                gen_seq = result["struct_sequence"]
                seq_id = round(sum(a == b for a, b in zip(gen_seq, wt_sequence)) / L, 4)

                coords_aligned = global_align(coords, wt_coords, all_positions)

                record = {
                    "sample_id": sample_id,
                    "seq_identity": seq_id,
                    "rmsd_fixed": None,
                    "rmsd_global": None,
                    "lddt_fixed": None,
                    "lddt_global": None,
                    "protein": result["struct_protein"],
                }

                if coords_aligned is not None:
                    record["rmsd_fixed"]  = compute_rmsd(coords_aligned, wt_coords, fixed_set)
                    record["rmsd_global"] = compute_rmsd(coords_aligned, wt_coords, all_positions)

                record["lddt_fixed"]  = compute_lddt_ca(coords, wt_coords, fixed_set)
                record["lddt_global"] = compute_lddt_ca(coords, wt_coords, all_positions)

                write(f"    Seq identity:    {record['seq_identity']}")
                write(f"    Fixed-site RMSD: {record['rmsd_fixed']} A")
                write(f"    Global RMSD:     {record['rmsd_global']} A")
                write(f"    Fixed-site lDDT: {record['lddt_fixed']}")
                write(f"    Global lDDT:     {record['lddt_global']}")

                records.append(record)

            # ---- Summary ----
            valid = [r for r in records if r["rmsd_global"] is not None]
            write(f"\n  [Random Set {set_idx + 1}] Summary ({len(valid)}/{N_SAMPLES} valid):")

            if valid:
                for key, label in [
                    ("rmsd_global",  "Mean global RMSD"),
                    ("rmsd_fixed",   "Mean fixed-site RMSD"),
                    ("lddt_global",  "Mean global lDDT-CA"),
                    ("lddt_fixed",   "Mean fixed-site lDDT-CA"),
                    ("seq_identity", "Mean seq identity"),
                ]:
                    vals = [r[key] for r in valid if r[key] is not None]
                    if vals:
                        m = sum(vals) / len(vals)
                        s = math.sqrt(sum((x - m)**2 for x in vals) / len(vals))
                        fmt = ".3f" if "rmsd" in key.lower() else ".4f"
                        unit = " A" if "rmsd" in key.lower() else ""
                        write(f"    {label:30s} {m:{fmt}} +/- {s:{fmt}}{unit}")

                # Save best PDB
                best = min(valid, key=lambda r: r["rmsd_global"])
                pdb_out = Path("pdbs_control") / f"random_set{set_idx+1}_best.pdb"
                try:
                    best["protein"].to_pdb(str(pdb_out))
                    write(f"  Best: sample {best['sample_id']+1}"
                          f" (global RMSD={best['rmsd_global']} A) -> {pdb_out}")
                except Exception as e:
                    write(f"  [Warning] Failed to save PDB: {e}")

        # ---- Reference: reprint G5 (core-only) summary for comparison ----
        write("")
        write("=" * 60)
        write("  Reference: G5 (catalytic core only) from main experiment")
        write(f"  Fixed residues (1-based): {sorted(core)}")
        write("  See results/results_CA2.txt for full G5 data")
        write("=" * 60)

    print(f"\n[Control] Results saved to: {output_path}")
    print("[Control] Done!")


if __name__ == "__main__":
    main()