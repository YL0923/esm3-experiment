# main.py
# Main entry point for gradient mask structure experiment
# Runs all proteins defined in config.PROTEINS in a single execution
#
# Usage:
#   python main.py

import math
from pathlib import Path

from config import PROTEINS, CONDITIONS, N_SAMPLES
from structure import load_pdb, build_prompt, compute_rmsd, compute_lddt_ca
from runner import load_model, run_one_sample


def run_protein(protein_cfg: dict, model, conditions: list, n_samples: int):
    """Run the full experiment for one protein."""
    pname      = protein_cfg["name"]
    pdb_path   = protein_cfg["pdb_path"]
    pdb_chain  = protein_cfg["pdb_chain"]
    out_file   = protein_cfg["output_file"]
    struct_dir = protein_cfg["structure_dir"]
    seq_length = protein_cfg["seq_length"]

    # Build layer map from config
    layer_1 = protein_cfg["layer_1"]
    layer_2 = protein_cfg["layer_2"]
    layer_map = {i: protein_cfg[f"layer_{i}"] for i in range(1, 7)}

    print("\n" + "#" * 60)
    print(f"#  Protein: {pname}")
    print(f"#  PDB: {pdb_path} chain {pdb_chain}")
    print("#" * 60)

    # ---- Prepare output ----
    Path(struct_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(out_file)

    # ---- Load WT structure ----
    wt_protein  = load_pdb(pdb_path, pdb_chain)
    wt_coords   = wt_protein.coordinates
    wt_sequence = wt_protein.sequence
    L = len(wt_sequence)
    assert L == seq_length, f"{pname}: sequence length {L} != expected {seq_length}"

    all_results = []

    with open(output_path, 'w', encoding='utf-8') as out:

        def write(line=""):
            print(line)
            out.write(line + '\n')

        write("=" * 60)
        write(f"  {pname} Gradient Mask Structure Experiment Results")
        write("=" * 60)
        write(f"WT sequence length: {L}")
        write(f"WT sequence: {wt_sequence}")

        for condition in conditions:
            cond_name   = condition["name"]
            cond_label  = condition["label"]
            mask_layers = condition["mask_layers"]
            fixed_mode  = condition.get("fixed_mode", "core_shell")

            write("")
            write("=" * 60)
            write(f"Condition: {cond_label}")
            write(f"Masked layers: {mask_layers}")
            write(f"Fixed mode:    {fixed_mode}")
            write("=" * 60)

            prompt = build_prompt(
                wt_protein, mask_layers, fixed_mode,
                layer_map, layer_1, layer_2,
            )
            group_records = []

            for sample_id in range(n_samples):
                write(f"\n  --- Sample {sample_id + 1}/{n_samples} ---")

                result = run_one_sample(model, prompt, sample_id)

                record = {
                    "condition_name":  cond_name,
                    "condition_label": cond_label,
                    "mask_layers":     mask_layers,
                    "sample_id":       sample_id,
                    "sequence":        None,
                    "seq_identity":    None,
                    "rmsd_layer1":     None,
                    "rmsd_constrained": None,
                    "rmsd_global":     None,
                    "lddt_layer1":     None,
                    "lddt_global":     None,
                    "error":           None,
                    "protein":         None,
                }

                if result is None:
                    record["error"] = "generation failed"
                    write("  Generation failed")
                else:
                    coords  = result["struct_protein"].coordinates
                    gen_seq = result["struct_sequence"]
                    record["sequence"]     = gen_seq
                    record["protein"]      = result["struct_protein"]
                    record["seq_identity"] = round(
                        sum(a == b for a, b in zip(gen_seq, wt_sequence)) / L, 4
                    )
                    record["rmsd_layer1"]      = compute_rmsd(coords, wt_coords, layer_1)
                    record["rmsd_constrained"] = compute_rmsd(coords, wt_coords, layer_1 | layer_2)
                    record["rmsd_global"]      = compute_rmsd(
                        coords, wt_coords, set(range(1, L + 1))
                    )
                    record["lddt_layer1"]  = compute_lddt_ca(coords, wt_coords, layer_1)
                    record["lddt_global"]  = compute_lddt_ca(
                        coords, wt_coords, set(range(1, L + 1))
                    )

                    write(f"    Sequence:              {gen_seq}")
                    write(f"    Seq identity to WT:    {record['seq_identity']}")
                    write(f"    Catalytic core RMSD:   {record['rmsd_layer1']} A")
                    write(f"    Constrained site RMSD: {record['rmsd_constrained']} A")
                    write(f"    Global RMSD:           {record['rmsd_global']} A")
                    write(f"    Catalytic core lDDT:   {record['lddt_layer1']}")
                    write(f"    Global lDDT:           {record['lddt_global']}")

                group_records.append(record)
                all_results.append(record)

            # ---- Group summary + save best PDB ----
            valid = [r for r in group_records if r["rmsd_global"] is not None]

            write(f"\n  [{cond_label}] Summary:")
            write(f"  Valid samples: {len(valid)}/{n_samples}")

            if valid:
                globals_rmsd     = [r["rmsd_global"] for r in valid]
                cores_rmsd       = [r["rmsd_layer1"] for r in valid if r["rmsd_layer1"] is not None]
                constrained_rmsd = [r["rmsd_constrained"] for r in valid if r["rmsd_constrained"] is not None]
                seq_ids          = [r["seq_identity"] for r in valid if r["seq_identity"] is not None]

                avg_g = sum(globals_rmsd) / len(globals_rmsd)
                std_g = math.sqrt(sum((x - avg_g)**2 for x in globals_rmsd) / len(globals_rmsd))

                lddt_g = [r["lddt_global"] for r in valid if r["lddt_global"] is not None]
                lddt_c = [r["lddt_layer1"] for r in valid if r["lddt_layer1"] is not None]

                write(f"    Mean global RMSD:           {avg_g:.3f} +/- {std_g:.3f} A")
                if cores_rmsd:
                    avg_c = sum(cores_rmsd) / len(cores_rmsd)
                    std_c = math.sqrt(sum((x - avg_c)**2 for x in cores_rmsd) / len(cores_rmsd))
                    write(f"    Mean catalytic core RMSD:   {avg_c:.3f} +/- {std_c:.3f} A")
                if constrained_rmsd:
                    avg_cs = sum(constrained_rmsd) / len(constrained_rmsd)
                    std_cs = math.sqrt(sum((x - avg_cs)**2 for x in constrained_rmsd) / len(constrained_rmsd))
                    write(f"    Mean constrained site RMSD: {avg_cs:.3f} +/- {std_cs:.3f} A")
                if lddt_g:
                    avg_lg = sum(lddt_g) / len(lddt_g)
                    std_lg = math.sqrt(sum((x - avg_lg)**2 for x in lddt_g) / len(lddt_g))
                    write(f"    Mean global lDDT-CA:        {avg_lg:.4f} +/- {std_lg:.4f}")
                if lddt_c:
                    avg_lc = sum(lddt_c) / len(lddt_c)
                    std_lc = math.sqrt(sum((x - avg_lc)**2 for x in lddt_c) / len(lddt_c))
                    write(f"    Mean core lDDT-CA:          {avg_lc:.4f} +/- {std_lc:.4f}")
                if seq_ids:
                    avg_si = sum(seq_ids) / len(seq_ids)
                    std_si = math.sqrt(sum((x - avg_si)**2 for x in seq_ids) / len(seq_ids))
                    write(f"    Mean seq identity to WT:    {avg_si:.4f} +/- {std_si:.4f}")

                # Save best PDB
                best = min(valid, key=lambda r: r["rmsd_global"])
                pdb_out = Path(struct_dir) / f"{cond_name}_best.pdb"
                try:
                    best["protein"].to_pdb(str(pdb_out))
                    write(f"  Best: sample {best['sample_id']+1}"
                          f" (global RMSD={best['rmsd_global']} A) -> {pdb_out}")
                except Exception as e:
                    write(f"  [Warning] Failed to save PDB: {e}")

        # ---- Final summary ----
        write("")
        write("=" * 60)
        write(f"  {pname} Final Summary")
        write("=" * 60)
        for condition in conditions:
            cond_name  = condition["name"]
            cond_label = condition["label"]
            valid = [
                r for r in all_results
                if r["condition_name"] == cond_name
                and r["rmsd_global"] is not None
            ]
            write(f"\n  {cond_label} ({len(valid)} valid samples)")
            if valid:
                globals_rmsd     = [r["rmsd_global"] for r in valid]
                cores_rmsd       = [r["rmsd_layer1"] for r in valid if r["rmsd_layer1"] is not None]
                constrained_rmsd = [r["rmsd_constrained"] for r in valid if r["rmsd_constrained"] is not None]
                lddt_g = [r["lddt_global"] for r in valid if r["lddt_global"] is not None]
                lddt_c = [r["lddt_layer1"] for r in valid if r["lddt_layer1"] is not None]
                seq_ids = [r["seq_identity"] for r in valid if r["seq_identity"] is not None]
                write(f"    Mean global RMSD:           {sum(globals_rmsd)/len(globals_rmsd):.3f} A")
                if cores_rmsd:
                    write(f"    Mean catalytic core RMSD:   {sum(cores_rmsd)/len(cores_rmsd):.3f} A")
                if constrained_rmsd:
                    write(f"    Mean constrained site RMSD: {sum(constrained_rmsd)/len(constrained_rmsd):.3f} A")
                write(f"    Best global RMSD:           {min(globals_rmsd):.3f} A")
                if lddt_g:
                    write(f"    Mean global lDDT-CA:        {sum(lddt_g)/len(lddt_g):.4f}")
                if lddt_c:
                    write(f"    Mean core lDDT-CA:          {sum(lddt_c)/len(lddt_c):.4f}")
                if seq_ids:
                    write(f"    Mean seq identity to WT:    {sum(seq_ids)/len(seq_ids):.4f}")
        write("=" * 60)

    print(f"\n[{pname}] Results saved to: {output_path}")
    print(f"[{pname}] Best PDBs saved to: {struct_dir}/")

    # ---- Generate plots ----
    plot_results(all_results, pname)


def plot_results(all_results: list, protein_name: str):
    """Generate RMSD and lDDT trend plots for one protein."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = []
    seen = set()
    for r in all_results:
        name = r["condition_name"]
        if name not in seen:
            seen.add(name)
            conditions.append((name, r["condition_label"]))

    short_labels = {
        "mask_layer6":     "G1\n(L6)",
        "mask_layer5_6":   "G2\n(L5-6)",
        "mask_layer4_5_6": "G3\n(L4-6)",
        "mask_layer3_6":   "G4\n(L3-6)",
        "mask_layer2_6":   "G5\n(L2-6)",
    }

    def _gather(records, key):
        vals = [r[key] for r in records if r.get(key) is not None]
        if not vals:
            return None, None
        mean = sum(vals) / len(vals)
        std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        return mean, std

    x_labels = []
    global_rmsd, global_rmsd_err = [], []
    core_rmsd, core_rmsd_err = [], []
    global_lddt, global_lddt_err = [], []

    for cond_name, _ in conditions:
        x_labels.append(short_labels.get(cond_name, cond_name))
        recs = [r for r in all_results if r["condition_name"] == cond_name]
        m, s = _gather(recs, "rmsd_global")
        global_rmsd.append(m); global_rmsd_err.append(s)
        m, s = _gather(recs, "rmsd_layer1")
        core_rmsd.append(m); core_rmsd_err.append(s)
        m, s = _gather(recs, "lddt_global")
        global_lddt.append(m); global_lddt_err.append(s)

    x = np.arange(len(x_labels))

    def _safe_plot(ax, x, means, errs, **kwargs):
        valid = [(xi, m, e) for xi, m, e in zip(x, means, errs) if m is not None]
        if not valid:
            return
        xv, mv, ev = zip(*valid)
        ax.errorbar(list(xv), list(mv), yerr=list(ev), **kwargs)

    # Figure 1: RMSD (global + core)
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    _safe_plot(ax1, x, global_rmsd, global_rmsd_err,
               fmt='o-', capsize=4, label='Global Backbone RMSD',
               color='#2196F3', linewidth=2, markersize=7)
    _safe_plot(ax1, x, core_rmsd, core_rmsd_err,
               fmt='s--', capsize=4, label='Core cRMSD (Layer 1)',
               color='#E53935', linewidth=2, markersize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel('Masking Condition')
    ax1.set_ylabel('Backbone RMSD (Å)')
    ax1.set_title(f'{protein_name}: Backbone RMSD vs. Masking Extent')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fname1 = f"fig1_rmsd_{protein_name}.png"
    fig1.savefig(fname1, dpi=150)
    plt.close(fig1)
    print(f"[Plot] Saved {fname1}")

    # Figure 2: lDDT
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    _safe_plot(ax2, x, global_lddt, global_lddt_err,
               fmt='o-', capsize=4, label='Global lDDT-CA',
               color='#2196F3', linewidth=2, markersize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel('Masking Condition')
    ax2.set_ylabel('Global lDDT-CA')
    ax2.set_title(f'{protein_name}: lDDT-CA vs. Masking Extent')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fname2 = f"fig2_lddt_{protein_name}.png"
    fig2.savefig(fname2, dpi=150)
    plt.close(fig2)
    print(f"[Plot] Saved {fname2}")


def main():
    print("=" * 60)
    print("  Gradient Mask Structure Experiment")
    print(f"  Proteins: {', '.join(p['name'] for p in PROTEINS)}")
    print("=" * 60)

    # Load model once, share across all proteins
    model = load_model()

    for protein_cfg in PROTEINS:
        run_protein(protein_cfg, model, CONDITIONS, N_SAMPLES)

    print("\n" + "=" * 60)
    print("  All proteins completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()