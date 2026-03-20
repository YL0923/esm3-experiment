# CA2 Structural Plasticity Experiment with ESM3

## Overview

This project investigates how ESM3 (1.4B) reconstructs the structure of Human Carbonic Anhydrase II (CA2, PDB: 1CA2) when sequence and structural information is progressively removed from the periphery while the catalytic core is kept fixed.

## Motivation

CA2 has a well-defined active site centered on a zinc ion coordinated by three histidines (His94, His96, His119), making it a clean test case for studying how much contextual information is needed to maintain structural integrity around a functional site.

## Scientific Question

> When the catalytic core residues are fixed, how does progressive masking of surrounding regions affect ESM3's ability to reconstruct the native CA2 structure?

## Residue Layering

Residues are divided into four spatial layers based on Cα distance to the centroid of the three zinc-coordinating histidines (His94, His96, His119):

| Layer | Distance | Residues | Role in Experiment |
|-------|----------|----------|--------------------|
| Layer 1 | Catalytic core | 7 | Always fixed |
| Layer 2 | Functional shell | 8 | Always fixed |
| Layer 3 | Middle region (<20Å) | 171 | Masked in Condition 2 |
| Layer 4 | Distal region (≥20Å) | 70 | Masked in all conditions |

**Layer 1** (defined by biological function):
His64, His94, His96, Glu106, His119, Thr199, Thr200

**Layer 2** (functional shell residues known to stabilize the active site):
Tyr7, Asn62, Asn67, Val121, Val143, Leu198, Val207, Trp209

**Layers 3 and 4** are defined by spatial distance from the catalytic centroid, with 20Å as the threshold.

## Experimental Conditions

| Condition | Masked Layers | Free Residues |
|-----------|--------------|---------------|
| Condition 1 | Layer 4 only | 70 |
| Condition 2 | Layers 3 + 4 | 241 |

In both conditions:
- Fixed residues (Layers 1+2): sequence identity and backbone coordinates provided to ESM3
- Masked residues: sequence set to `_`, coordinates set to `None`

## Generation Pipeline

For each sample:

```
Prompt (fixed sequence + fixed coordinates)
    → Step 1: Generate structure (coordinate-guided)
    → Step 2: Generate sequence (fill masked positions)
    → Kabsch superimposition onto 1CA2
    → Compute RMSD at three levels
```

RMSD is computed after Kabsch superimposition at three levels:
- **Layer 1 RMSD**: catalytic core only
- **Layer 2 RMSD**: functional shell only
- **Global RMSD**: all 256 residues (primary metric)

## Preliminary Results (3 samples per condition, subject to change)

| Condition | Global RMSD | Core RMSD | Shell RMSD |
|-----------|------------|-----------|------------|
| Condition 1 (mask layer 4) | 0.629 Å | 0.460 Å | 0.296 Å |
| Condition 2 (mask layers 3+4) | 2.563 Å | 0.780 Å | 0.870 Å |

## Planned Next Steps

- Add intermediate gradient conditions (subdivide Layer 3 into 14–17Å and 17–20Å)
- Add a condition where Layer 2 (functional shell) is also masked
- Increase N_SAMPLES to 16 for statistical reliability
- Visualize structures in PyMOL

## File Structure

```
├── config.py       # All parameters, layer definitions, experimental conditions
├── structure.py    # PDB loading, prompt construction, Kabsch RMSD
├── runner.py       # ESM3 model loading and generation
├── main.py         # Main loop, result collection and output
└── 1CA2.pdb        # Input structure (download separately)
```

## Usage

**Download 1CA2:**
```bash
wget https://files.rcsb.org/download/1CA2.pdb
```

**Run experiment:**
```bash
python main.py
```

**Output:**
- `gradient_mask_results.txt`: all results in human-readable format
- `gradient_mask_pdbs/`: best PDB structure per condition (lowest global RMSD)

## Key Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_SAMPLES` | 16 | Samples per condition |
| `STRUCT_TEMPERATURE` | 0.7 | Structure generation temperature |
| `SEQ_TEMPERATURE` | 0.7 | Sequence generation temperature |
| `STRUCT_NUM_STEPS` | 8 | Iterative decoding steps |

## Dependencies

```
esm  (evolutionaryscale)
torch
```

## Notes

- Residue numbering follows the 256-aa visible sequence extracted from 1CA2 (1-indexed), not the original PDB numbering (which starts at 4)
- Model used: `esm3-sm-open-v1` (1.4B parameters)
- RMSD values are computed after Kabsch superimposition
