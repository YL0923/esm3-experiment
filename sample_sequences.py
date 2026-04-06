"""
sample_sequences.py
--------------------
Randomly sample N sequences per condition from each protein's FASTA file
and write the sampled sequences to new FASTA files for ColabFold validation.

Usage:
    python sample_sequences.py

Input files (place in the same directory):
    sequences_CA2.fasta
    sequences_CA9.fasta
    sequences_CPA.fasta

Output files (written to sampled/):
    sampled_CA2.fasta
    sampled_CA9.fasta
    sampled_CPA.fasta
"""

import random
from collections import defaultdict
from pathlib import Path

# ============================================================
# Parameters
# ============================================================
N_SAMPLES = 5       # number of sequences to sample per condition
SEED = 42           # fixed random seed for reproducibility

PROTEINS = ["CA2", "CA9", "CPA"]
OUTPUT_DIR = Path("sample_sequences")

CONDITIONS = [
    "mask_layer6",
    "mask_layer5_6",
    "mask_layer4_5_6",
    "mask_layer3_4_5_6",
    "mask_layer2_3_4_5_6",
]

# ============================================================
# Parse FASTA file
# ============================================================

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (header, sequence) tuples."""
    records = []
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            elif line:
                seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def get_condition(header: str, protein: str) -> str | None:
    """
    Extract condition name from a FASTA header.
    Example: CA2_mask_layer6_s1 -> mask_layer6
    """
    prefix = protein + "_"
    if not header.startswith(prefix):
        return None
    rest = header[len(prefix):]           # e.g. mask_layer6_s1
    # strip trailing _sN
    parts = rest.rsplit("_s", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0]                        # e.g. mask_layer6


# ============================================================
# Main
# ============================================================

rng = random.Random(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

for protein in PROTEINS:
    fasta_path = Path(f"sequences/sequences_{protein}.fasta")
    if not fasta_path.exists():
        print(f"[Skip] {fasta_path} not found")
        continue

    records = parse_fasta(fasta_path)

    # Group sequences by condition
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for header, seq in records:
        cond = get_condition(header, protein)
        if cond:
            groups[cond].append((header, seq))

    # Sample N sequences per condition
    sampled = []
    for cond in CONDITIONS:
        pool = groups.get(cond, [])
        if not pool:
            print(f"[Warn] {protein} / {cond}: no sequences found")
            continue
        n = min(N_SAMPLES, len(pool))
        chosen = rng.sample(pool, n)
        sampled.extend(chosen)
        print(f"[{protein}] {cond}: sampled {n}/{len(pool)} sequences "
              f"-> {[h.split('_s')[-1] for h, _ in chosen]}")

    # Write output
    out_path = OUTPUT_DIR / f"sampled_{protein}.fasta"
    with open(out_path, "w") as f:
        for header, seq in sampled:
            f.write(f">{header}\n{seq}\n")

    print(f"[Done] {protein}: {len(sampled)} sequences -> {out_path}\n")