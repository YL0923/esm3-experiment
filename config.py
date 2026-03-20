# config.py
# CA2 carbonic anhydrase gradient mask experiment configuration
#
# Residue numbering:
#   Sequence index: numbering used in this experiment, 1-indexed, 1 to 256, continuous without gaps
#   PDB numbering: original numbering in 1CA2, from 4 to 260, with missing residues (first 3 residues absent)
#   Mapping: sequence index = order of residues with coordinates in the PDB
#
# Layer definition:
#   Layer 1: catalytic core (7 residues, manually defined, biologically meaningful)
#   Layer 2: functional shell (8 residues, manually defined, surrounding the active site)
#   Layer 3: intermediate region (171 residues, within <20Å from catalytic center, excluding Layer 1 and 2)
#   Layer 4: distal region (70 residues, ≥20Å from catalytic center)

# ============================================================
# 1. Model and file paths
# ============================================================
MODEL_NAME    = "esm3-sm-open-v1"
DEVICE        = "cuda"
PDB_PATH      = "1CA2.pdb"
PDB_CHAIN     = "A"
OUTPUT_FILE   = "gradient_mask_results.jsonl"
STRUCTURE_DIR = "gradient_mask_pdbs"

# ============================================================
# 2. Layer definitions (sequence indices, 1-indexed)
# ============================================================

# Layer 1: catalytic core (always fixed, sequence identity + coordinates not released)
# PDB: His64, His94, His96, Glu106, His119, Thr199, Thr200
LAYER_1 = {61, 91, 93, 103, 116, 195, 196}

# Layer 2: functional shell (always fixed, sequence identity + coordinates not released)
# PDB: Tyr7, Asn62, Asn67, Val121, Val143, Leu198, Val207, Trp209
LAYER_2 = {4, 59, 64, 118, 139, 194, 203, 205}

# Layer 3: intermediate region (distance to catalytic center < 20Å, excluding Layer 1 and 2)
LAYER_3 = {
    2, 3, 5, 8, 9, 10, 13, 20, 24, 25, 26, 27, 28, 29, 30, 44, 46, 47,
    48, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 65, 66, 67, 68, 69,
    74, 75, 76, 85, 86, 87, 88, 89, 90, 92, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 117,
    119, 120, 121, 127, 128, 130, 131, 136, 137, 138, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
    172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
    186, 187, 188, 189, 190, 191, 192, 193, 197, 198, 199, 200, 201, 202,
    204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
    219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 235, 236, 237, 238,
    239, 240, 241, 242, 243, 244, 245
}

# Layer 4: distal region (distance to catalytic center ≥ 20Å)
LAYER_4 = {
    1, 6, 7, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 49, 50, 70, 71, 72, 73, 77,
    78, 79, 80, 81, 82, 83, 84, 107, 122, 123, 124, 125, 126, 129, 132,
    133, 134, 135, 229, 230, 231, 232, 233, 234, 246, 247, 248, 249, 250,
    251, 252, 253, 254, 255, 256
}

assert len(LAYER_1) + len(LAYER_2) + len(LAYER_3) + len(LAYER_4) == 256

# Layer 1 + Layer 2: always fixed (both sequence identity and coordinates are preserved)
FIXED_RESIDUES = LAYER_1 | LAYER_2   # total 15 residues

# ============================================================
# 3. Experiment condition definitions
# Each condition defines which layers are masked (both sequence and coordinates released, Scheme A)
# FIXED_RESIDUES remain unchanged in all conditions
# ============================================================

CONDITIONS = [
    {
        "name":        "mask_layer4",
        "label":       "Mask distal only (Layer 4)",
        "mask_layers": [4],
        "no_coords":   False,   # keep coordinate constraints
    },
    {
        "name":        "mask_layer3_4",
        "label":       "Mask intermediate + distal (Layer 3, 4)",
        "mask_layers": [3, 4],
        "no_coords":   False,   # keep coordinate constraints
    },
    {
        "name":        "seq_only_control",
        "label":       "Control: sequence-only, no coordinates",
        "mask_layers": [3, 4],  # same mask range as condition 2 for direct comparison
        "no_coords":   True,    # remove all coordinates
    },
]

# ============================================================
# 4. Generation parameters
# ============================================================
N_SAMPLES          = 3
STRUCT_NUM_STEPS   = 8
STRUCT_TEMPERATURE = 0.7

# Mapping from layer index to residue sets (used in structure.py)
LAYER_MAP = {1: LAYER_1, 2: LAYER_2, 3: LAYER_3, 4: LAYER_4}

SEQ_NUM_STEPS   = 8
SEQ_TEMPERATURE = 0.7