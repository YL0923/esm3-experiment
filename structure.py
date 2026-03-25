# structure.py
# Load PDB structure, build ESMProtein prompt, compute RMSD and lDDT

import torch
from pathlib import Path
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain

CA_IDX = 1              # CA index in atom37 format (used by lDDT)
BACKBONE_IDX = [0, 1, 2]  # N, CA, C in atom37 format (O excluded: not always present in ESM3 decoder output)


def load_pdb(pdb_path: str, chain: str) -> ESMProtein:
    """Load structure from PDB file, return ESMProtein object."""
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    chain_obj = ProteinChain.from_pdb(str(pdb_path), chain_id=chain)
    protein = ESMProtein.from_protein_chain(chain_obj)
    print(f"[PDB] Sequence length: {len(protein.sequence)}")
    print(f"[PDB] Coordinates shape: {protein.coordinates.shape}")
    return protein


def get_fixed_residues(fixed_mode: str, layer_1: set, layer_2: set) -> set:
    """
    Decide which residues are fixed for a condition.

    fixed_mode:
        - "core_shell": keep layer 1 + layer 2 fixed
        - "core_only" : keep only layer 1 fixed
    """
    if fixed_mode == "core_only":
        return set(layer_1)
    if fixed_mode == "core_shell":
        return set(layer_1 | layer_2)
    raise ValueError(f"Unknown fixed_mode: {fixed_mode}")


def build_prompt(wt_protein: ESMProtein,
                 mask_layers: list,
                 fixed_mode: str,
                 layer_map: dict,
                 layer_1: set,
                 layer_2: set) -> ESMProtein:
    """
    Build ESMProtein prompt:
    - fixed residues: sequence and coordinates retained
    - masked layers: sequence set to '_', coordinates set to nan
    """
    wt_seq    = wt_protein.sequence
    wt_coords = wt_protein.coordinates.clone()
    L = len(wt_seq)

    fixed_residues = get_fixed_residues(fixed_mode, layer_1, layer_2)

    # Determine residues to mask
    mask_positions = set()
    for layer_id in mask_layers:
        layer_set = layer_map[layer_id]
        mask_positions |= (layer_set - fixed_residues)

    # Build sequence prompt
    seq_list = list(wt_seq)
    for pos_1based in mask_positions:
        seq_list[pos_1based - 1] = '_'
    prompt_seq = ''.join(seq_list)

    # Build coordinates (set masked positions to nan)
    prompt_coords = wt_coords.clone()
    for pos_1based in mask_positions:
        prompt_coords[pos_1based - 1] = float('nan')

    n_masked = prompt_seq.count('_')
    print(
        f"[Prompt] fixed_mode={fixed_mode} | "
        f"Fixed: {L - n_masked} | Masked: {n_masked} | Total: {L}"
    )

    return ESMProtein(sequence=prompt_seq, coordinates=prompt_coords)


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Kabsch algorithm: align P onto Q, return aligned P.
    P, Q shape: [N, 3]
    """
    p_center = P.mean(dim=0)
    q_center = Q.mean(dim=0)
    P_c = P - p_center
    Q_c = Q - q_center

    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)

    d = torch.linalg.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype, device=P.device))
    R = Vt.T @ D @ U.T

    return (P_c @ R.T) + q_center


def compute_rmsd(coords_gen: torch.Tensor,
                 coords_wt: torch.Tensor,
                 positions: set) -> float | None:
    """
    Compute backbone RMSD (N, CA, C, O) with Kabsch alignment.
    """
    gen_bb, wt_bb = [], []
    for pos_1based in sorted(positions):
        idx = pos_1based - 1
        if idx >= coords_gen.shape[0] or idx >= coords_wt.shape[0]:
            continue
        g = coords_gen[idx, BACKBONE_IDX, :]
        w = coords_wt[idx, BACKBONE_IDX, :]
        if torch.isnan(g).any() or torch.isnan(w).any():
            continue
        gen_bb.append(g)
        wt_bb.append(w)

    if len(gen_bb) < 3:
        return None

    gen_bb = torch.cat(gen_bb, dim=0).float()
    wt_bb  = torch.cat(wt_bb, dim=0).float()

    # Safety check: SVD will fail on non-finite values
    if not torch.isfinite(gen_bb).all() or not torch.isfinite(wt_bb).all():
        return None

    gen_aligned = kabsch_align(gen_bb, wt_bb)
    diff = gen_aligned - wt_bb
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()
    return round(rmsd, 3)


def compute_lddt_ca(coords_gen: torch.Tensor,
                    coords_wt: torch.Tensor,
                    positions: set,
                    cutoff: float = 15.0,
                    thresholds: tuple = (0.5, 1.0, 2.0, 4.0)) -> float | None:
    """
    Compute lDDT-CA (no alignment needed).
    """
    pos_list = sorted(positions)
    gen_ca, wt_ca = [], []
    for pos_1based in pos_list:
        idx = pos_1based - 1
        if idx >= coords_gen.shape[0] or idx >= coords_wt.shape[0]:
            continue
        g = coords_gen[idx, CA_IDX, :]
        w = coords_wt[idx, CA_IDX, :]
        if torch.isnan(g).any() or torch.isnan(w).any():
            continue
        gen_ca.append(g)
        wt_ca.append(w)

    N = len(gen_ca)
    if N < 2:
        return None

    gen_ca = torch.stack(gen_ca).float()
    wt_ca  = torch.stack(wt_ca).float()

    dist_gen = torch.cdist(gen_ca, gen_ca)
    dist_wt  = torch.cdist(wt_ca, wt_ca)

    scores = []
    for i in range(N):
        mask = (dist_wt[i] < cutoff) & (torch.arange(N) != i)
        if mask.sum() == 0:
            continue
        diff = torch.abs(dist_gen[i, mask] - dist_wt[i, mask])
        preserved = sum((diff < t).float().mean().item() for t in thresholds)
        scores.append(preserved / len(thresholds))

    if not scores:
        return None

    lddt = sum(scores) / len(scores)
    return round(lddt, 4)