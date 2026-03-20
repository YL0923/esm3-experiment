# main.py
# CA2 gradient mask structure experiment main entry
#
# Usage:
#   python main.py
#
# Output:
#   - gradient_mask_results.txt: all experiment results
#   - gradient_mask_pdbs/: PDB files with lowest RMSD per group (one per group)

import json
from pathlib import Path

from config import (
    CONDITIONS, N_SAMPLES,
    LAYER_1, LAYER_2,
    OUTPUT_FILE, STRUCTURE_DIR,
)
from structure import load_pdb, build_prompt, compute_rmsd
from runner import load_model, run_one_sample


def main():
    print("=" * 60)
    print("  CA2 Gradient Mask Structure Experiment")
    print("=" * 60)

    # ---- Prepare output ----
    Path(STRUCTURE_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_FILE)
    # Replace jsonl with txt for easier reading
    output_path = output_path.with_suffix('.txt')

    # ---- Load WT structure ----
    print("\n[Step 1/3] Loading 1CA2 wild-type structure...")
    wt_protein  = load_pdb()
    wt_coords   = wt_protein.coordinates
    wt_sequence = wt_protein.sequence
    L = len(wt_sequence)

    # ---- Load model ----
    print("\n[Step 2/3] Loading ESM3 model...")
    model = load_model()

    # ---- Run experiments ----
    print("\n[Step 3/3] Running experiments...")
    all_results = []

    with open(output_path, 'w', encoding='utf-8') as out:

        def write(line=""):
            """Write to file and print to console simultaneously"""
            print(line)
            out.write(line + '\n')

        write("=" * 60)
        write("  CA2 Gradient Mask Structure Experiment Results")
        write("=" * 60)
        write(f"WT sequence length: {L}")
        write(f"WT sequence: {wt_sequence}")

        for condition in CONDITIONS:
            cond_name   = condition["name"]
            cond_label  = condition["label"]
            mask_layers = condition["mask_layers"]

            write("")
            write("=" * 60)
            write(f"Condition: {cond_label}")
            write(f"Mask layers: Layer {mask_layers}")
            write("=" * 60)

            no_coords = condition.get("no_coords", False)
            prompt = build_prompt(wt_protein, mask_layers, no_coords=no_coords)
            group_records = []

            for sample_id in range(N_SAMPLES):
                write(f"\n  --- Sample {sample_id + 1}/{N_SAMPLES} ---")

                result = run_one_sample(model, prompt, sample_id)

                record = {
                    "condition_name":  cond_name,
                    "condition_label": cond_label,
                    "mask_layers":     mask_layers,
                    "sample_id":       sample_id,
                    "sequence":        None,
                    "rmsd_layer1":     None,
                    "rmsd_layer2":     None,
                    "rmsd_global":     None,
                    "error":           None,
                    "protein":         None,  # temporarily store protein object for saving best PDB
                }

                if result is None:
                    record["error"] = "Generation failed"
                    write("  Generation failed")
                else:
                    gen_sequence = result["sequence"]
                    gen_coords   = result["protein"].coordinates

                    record["sequence"] = gen_sequence
                    record["protein"]  = result["protein"]

                    record["rmsd_layer1"] = compute_rmsd(
                        gen_coords, wt_coords, LAYER_1
                    )
                    record["rmsd_layer2"] = compute_rmsd(
                        gen_coords, wt_coords, LAYER_2
                    )
                    record["rmsd_global"] = compute_rmsd(
                        gen_coords, wt_coords,
                        set(range(1, L + 1))
                    )

                    write(f"  Generated sequence: {gen_sequence}")
                    write(f"  Catalytic core RMSD (Layer 1): {record['rmsd_layer1']} Å")
                    write(f"  Functional shell RMSD (Layer 2): {record['rmsd_layer2']} Å")
                    write(f"  Global RMSD:                    {record['rmsd_global']} Å")

                group_records.append(record)
                all_results.append(record)

            # ---- End of group: summary + save best PDB ----
            valid = [r for r in group_records if r["rmsd_global"] is not None]

            write(f"\n  [{cond_label}] Summary:")
            write(f"  Valid samples: {len(valid)}/{N_SAMPLES}")

            if valid:
                avg_global = sum(r["rmsd_global"] for r in valid) / len(valid)
                avg_core   = sum(r["rmsd_layer1"] for r in valid) / len(valid)
                write(f"  Average global RMSD:     {avg_global:.3f} Å")
                write(f"  Average catalytic core RMSD: {avg_core:.3f} Å")

                # Find sample with lowest global RMSD and save PDB
                best = min(valid, key=lambda r: r["rmsd_global"])
                pdb_path = Path(STRUCTURE_DIR) / f"{cond_name}_best.pdb"
                try:
                    best["protein"].to_pdb(str(pdb_path))
                    write(f"  Best sample: Sample {best['sample_id']+1}"
                          f" (global RMSD={best['rmsd_global']} Å)"
                          f" -> saved to {pdb_path}")
                except Exception as e:
                    write(f"  [Warning] Failed to save best PDB: {e}")

        # ---- Final summary ----
        write("")
        write("=" * 60)
        write("  Final Summary")
        write("=" * 60)
        for condition in CONDITIONS:
            cond_name  = condition["name"]
            cond_label = condition["label"]
            valid = [
                r for r in all_results
                if r["condition_name"] == cond_name
                and r["rmsd_global"] is not None
            ]
            write(f"\n  {cond_label} ({len(valid)} valid samples)")
            if valid:
                avg_global = sum(r["rmsd_global"] for r in valid) / len(valid)
                avg_core   = sum(r["rmsd_layer1"] for r in valid) / len(valid)
                avg_shell  = sum(r["rmsd_layer2"] for r in valid) / len(valid)
                best_global = min(r["rmsd_global"] for r in valid)
                write(f"    Average global RMSD:     {avg_global:.3f} Å")
                write(f"    Average catalytic core RMSD: {avg_core:.3f} Å")
                write(f"    Average functional shell RMSD: {avg_shell:.3f} Å")
                write(f"    Best global RMSD:        {best_global:.3f} Å")
        write("=" * 60)

    print(f"\n[Done] Results saved to: {output_path}")
    print(f"[Done] Best PDBs saved in: {STRUCTURE_DIR}/")


if __name__ == "__main__":
    main()