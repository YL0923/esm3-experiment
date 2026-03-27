# ESM3 Gradient Masking Structure Generation Experiment

## Research Question

When catalytic core residues are constrained and surrounding regions are progressively masked, how well can ESM3 reconstruct the three-dimensional structure of a protein? How faithfully does ESM3 follow structural constraints (cRMSD), and how does overall structure quality degrade with increasing masking?

## Experimental Design

### Core Idea

Using zinc metalloenzymes as test cases, protein residues are divided into 6 layers based on distance from the catalytic center. The catalytic core (Layer 1) and/or functional shell (Layer 2) retain their wild-type sequence and coordinates as structural constraints. Outer layers are progressively masked (sequence set to `_`, coordinates set to `nan`), and ESM3 regenerates the masked regions.

### Protein Selection

| Protein | PDB | Chain | Residues | Type | Rationale |
|---------|-----|-------|----------|------|-----------|
| Human CA II | 1CA2 | A | 256 | Zinc metalloenzyme (carbonic anhydrase) | Most studied CA isoform; benchmark target |
| Human CA IX | 5DVX | A | 260 | Zinc metalloenzyme (carbonic anhydrase) | Same family, ~35% seq identity to CA II; tumor biomarker; tests intra-family generalization |
| Bovine CPA | 5CPA | A | 307 | Zinc metalloenzyme (carboxypeptidase) | Different fold and function from CAs; same metal cofactor; classic zinc protease; tests cross-enzyme generalization within zinc metalloenzymes |

### Layer Definitions

**Layer 1 (Catalytic Core)**: Residues directly involved in catalysis, supported by literature.

| | CA II (1CA2) | CA IX (5DVX) | CPA (5CPA) |
|---|---|---|---|
| Zn-coordinating | His94, His96, His119 | His226, His228, His251 | His69, Glu72, His196 |
| Proton shuttle | His64 | His200 | — |
| Substrate orientation | Thr199 | Thr332 | — |
| Catalytic base | — | — | Glu270 |
| **Total** | **5 residues** | **5 residues** | **4 residues** |

**Layer 2 (Functional Shell)**: Active-site residues modulating catalytic efficiency.

| | CA II (1CA2) | CA IX (5DVX) | CPA (5CPA) |
|---|---|---|---|
| Residues | Tyr7, Asn62, Asn67, Val121, Val143, Leu198, Trp209 | Trp141, Gln203, Gln224, Thr333, Pro334 | Arg71, Arg127, Asn144, Arg145, Tyr248 |
| **Total** | **7 residues** | **5 residues** | **5 residues** |

**Layers 3–6**: Automatically assigned by Cα distance to catalytic centroid (mean Cα of Zn-coordinating residues), with thresholds at 14 / 17 / 20 Å.

### Experimental Conditions (5 groups, shared across all proteins)

| Group | Masked Layers | Fixed Mode | Description |
|-------|--------------|------------|-------------|
| G1 | Layer 6 (>20 Å) | core + shell | Mask distal region only |
| G2 | Layer 5–6 (>17 Å) | core + shell | Extend masking inward |
| G3 | Layer 4–6 (>14 Å) | core + shell | Mask mid-to-distal |
| G4 | Layer 3–6 | core + shell | Only core and shell retained |
| G5 | Layer 2–6 | core only | Only catalytic core retained |

### Generation Pipeline

```
WT structure → Build prompt (fixed residues retain seq + coords; masked residues set to _/nan)
             → ESM3 generate structure tokens (iterative sampling, 128 steps)
             → ESM3 generate sequence (iterative sampling, 128 steps)
             → Evaluate metrics
```

10 samples per condition (different random seeds). Model loaded once, shared across all proteins. Temperature = 0.7.

## Evaluation Metrics

| Metric | Description | Paper Equivalent |
|--------|-------------|-----------------|
| **Core cRMSD** | Backbone RMSD (N, Cα, C) at Layer 1 residues only. Measures how faithfully ESM3 follows the catalytic core constraint | cRMSD (constrained site RMSD) |
| **Constrained site RMSD** | Backbone RMSD at Layer 1+2 residues. Measures constraint adherence across all fixed residues | — |
| **Global RMSD** | Backbone RMSD across all residues. Measures overall structure quality | — |
| **Core lDDT-CA** | Cα distance conservation at Layer 1. Alignment-free, complements cRMSD | — |
| **Global lDDT-CA** | Cα distance conservation across all residues | LDDT-CA (tokenizer evaluation) |
| **Sequence identity** | Fraction of generated sequence matching WT | — |

Backbone RMSD uses N, Cα, C (3 atoms), consistent with ESM3 paper Appendix A.1.7.3.1. Alignment via Kabsch algorithm (ref. 107 in paper). O atom excluded as ESM3 structure token decoder does not reliably output it.

RMSD reference: < 1 Å near-perfect; 1–2 Å good; 2–3 Å acceptable; > 3 Å significant deviation. ESM3 paper uses cRMSD < 1.5 Å as success threshold for motif scaffolding.

## File Structure

```
esm3-experiment/
├── config.py        # Protein definitions (PROTEINS list), layers, conditions, generation params
├── structure.py     # PDB loading, prompt building, RMSD/lDDT computation (config-independent)
├── runner.py        # ESM3 model loading and single-sample generation (structure-first path)
├── main.py          # Entry point: loops over all proteins, runs experiment, outputs results + plots
├── 1CA2.pdb         # CA II crystal structure
├── 5DVX_A.pdb       # CA IX crystal structure chain A
├── 5CPA.pdb         # CPA crystal structure
```

## Usage

```bash
python main.py
```

Single run completes all 3 proteins (150 samples total). Outputs per protein:

```
results_{name}.txt           # Detailed per-sample results + group summaries
pdbs_{name}/                 # Best PDB per condition
fig1_rmsd_{name}.png         # RMSD trend plot (global + core cRMSD)
fig2_lddt_{name}.png         # lDDT trend plot
```

## Current Progress

- [x] Pipeline code complete (config / structure / runner / main)
- [x] CA II layer definitions (literature-based L1–2 + distance-based L3–6)
- [x] CA IX layer definitions (same approach, PDB-to-sequence mapping verified)
- [x] CPA layer definitions (same approach, 5CPA verified)
- [x] Multi-protein single-run architecture (model loaded once)
- [x] Backbone RMSD (N, Cα, C) + lDDT-CA + sequence identity metrics
- [x] Core cRMSD + constrained site RMSD (Layer 1+2) + global RMSD
- [x] Auto-generated per-protein trend plots (global RMSD + core cRMSD on same figure)
- [x] Initial run completed for CA2 + CA9 (num_steps=8)
- [x] Full run with all 3 proteins (num_steps=128, production setting)
- [ ] Results analysis and combined multi-protein comparison plots
- [ ] PyMOL structure overlay visualization
- [ ] AlphaFold validation (fold ESM3-generated sequences with AF, compare to WT)

## Key Technical Decisions

1. **Backbone RMSD uses N, Cα, C (3 atoms).** ESM3 paper Appendix A.1.7.3.1 defines backbone distance loss on these 3 atoms. O atom excluded as it is not reliably output by the structure token decoder.

2. **Layers 1–2 defined by biological function; Layers 3–6 by distance.** Ensures catalytic core definitions are functionally equivalent across proteins.

3. **Generation path is structure-first**: generate structure tokens → generate sequence. Consistent with ESM3 paper's motif scaffolding workflow.

4. **128 decoding steps, temperature 0.7.** Paper uses L steps (sequence length) for prompt consistency evaluation and L/2 for motif scaffolding. 128 steps is close to L/2 for our proteins (256–307 residues).

5. **cRMSD computed on fixed residues with Kabsch alignment.** Same methodology as ESM3 paper (reference 107: Kabsch 1976). Measures how faithfully the model follows structural constraints after decode.
