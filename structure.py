# structure.py
# Load structure from 1CA2 PDB, build ESMProtein prompt, compute RMSD

import torch
from pathlib import Path
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain

from config import (
    PDB_PATH, PDB_CHAIN,
    LAYER_1, LAYER_2, LAYER_3, LAYER_4,
    FIXED_RESIDUES,
)

CA_IDX = 1  # Confirmed: Cα is index 1 in atom37 format


def load_pdb(pdb_path: str = PDB_PATH, chain: str = PDB_CHAIN) -> ESMProtein:
    """Load structure from PDB file and return an ESMProtein object."""
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(
            f"PDB file not found: {pdb_path}\n"
            f"Please download: wget https://files.rcsb.org/download/1CA2.pdb"
        )
    chain_obj = ProteinChain.from_pdb(str(pdb_path), chain_id=chain)
    protein = ESMProtein.from_protein_chain(chain_obj)
    print(f"[PDB Load] Sequence length: {len(protein.sequence)}")
    print(f"[PDB Load] Coordinates shape: {protein.coordinates.shape}")
    return protein


def build_prompt(wt_protein: ESMProtein,
                 mask_layers: list,
                 no_coords: bool = False) -> ESMProtein:
    """
    Build ESMProtein prompt:
    - FIXED_RESIDUES: keep sequence and coordinates (if no_coords=True, coordinates are also cleared)
    - mask layers: set sequence to '_' and coordinates to nan
    - other positions: keep both sequence and coordinates

    Args:
        no_coords: If True, remove all coordinates (control group to test if the model relies on memory)
    """
    wt_seq    = wt_protein.sequence
    wt_coords = wt_protein.coordinates.clone()
    L = len(wt_seq)

    # Determine residues to mask
    mask_positions = set()
    for layer_id in mask_layers:
        layer_set = {1: LAYER_1, 2: LAYER_2, 3: LAYER_3, 4: LAYER_4}[layer_id]
        mask_positions |= (layer_set - FIXED_RESIDUES)

    # Build sequence prompt
    seq_list = list(wt_seq)
    for pos_1based in mask_positions:
        seq_list[pos_1based - 1] = '_'
    prompt_seq = ''.join(seq_list)

    # Build coordinates
    if no_coords:
        # Control group: no coordinates, only sequence constraints
        prompt_coords = None
        print(f"[Prompt Build] Control group: no coordinates, only sequence constraints")
    else:
        # Experimental group: set masked positions to nan, keep others
        prompt_coords = wt_coords.clone()
        for pos_1based in mask_positions:
            prompt_coords[pos_1based - 1] = float('nan')

    n_masked = prompt_seq.count('_')
    print(f"[Prompt Build] Fixed residues: {L - n_masked} | Masked residues: {n_masked} | Total length: {L}")

    return ESMProtein(sequence=prompt_seq, coordinates=prompt_coords)


def kabsch_align(P: torch.Tensor, Q: torch.Tensor):
    """
    Kabsch algorithm: align P onto Q, return aligned P.
    P, Q shape: [N, 3]
    """
    # Centering
    p_center = P.mean(dim=0)
    q_center = Q.mean(dim=0)
    P_c = P - p_center
    Q_c = Q - q_center

    # SVD to compute optimal rotation matrix
    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection case
    d = torch.linalg.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype, device=P.device))

    R = Vt.T @ D @ U.T

    # Aligned P
    P_aligned = (P_c @ R.T) + q_center
    return P_aligned


def compute_rmsd(coords_gen: torch.Tensor,
                 coords_wt: torch.Tensor,
                 positions: set) -> float | None:
    """
    Compute Cα RMSD between generated structure and WT at specified positions.
    First align using Kabsch algorithm, then compute RMSD.

    Args:
        coords_gen: generated protein coordinates, shape [L, 37, 3]
        coords_wt: WT coordinates, shape [L, 37, 3]
        positions: residue positions (1-indexed)

    Returns:
        RMSD after alignment (Å), or None if insufficient valid coordinates
    """
    gen_ca, wt_ca = [], []
    for pos_1based in sorted(positions):
        idx = pos_1based - 1
        if idx >= coords_gen.shape[0] or idx >= coords_wt.shape[0]:
            continue
        g = coords_gen[idx, CA_IDX, :]
        w = coords_wt[idx, CA_IDX, :]
        if torch.isnan(g).any() or torch.isnan(w).any():
            continue
        gen_ca.append(g)
        wt_ca.append(w)

    if len(gen_ca) < 3:
        return None

    gen_ca = torch.stack(gen_ca).float()  # [N, 3]
    wt_ca  = torch.stack(wt_ca).float()   # [N, 3]

    # Kabsch alignment
    gen_aligned = kabsch_align(gen_ca, wt_ca)

    # Compute RMSD
    diff = gen_aligned - wt_ca
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()
    return round(rmsd, 3)