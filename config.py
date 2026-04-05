# config.py
# Configuration for gradient mask structure experiment

# ============================================================
# 1. Model and generation parameters
# ============================================================
MODEL_NAME         = "esm3-sm-open-v1"
DEVICE             = "cuda"
N_SAMPLES          = 16
STRUCT_NUM_STEPS   = 16
STRUCT_TEMPERATURE = 0.7
SEQ_NUM_STEPS      = 1
SEQ_TEMPERATURE    = 0.7
BASE_SEED          = 42

# ============================================================
# 2. Shared experimental conditions (same for all proteins)
# ============================================================
CONDITIONS = [
    {
        "name":        "mask_layer6",
        "label":       "Group 1: mask distal only (layer 6, >20A)",
        "mask_layers": [6],
        "fixed_mode":  "core_shell",
    },
    {
        "name":        "mask_layer5_6",
        "label":       "Group 2: mask layer 5+6 (>17A)",
        "mask_layers": [5, 6],
        "fixed_mode":  "core_shell",
    },
    {
        "name":        "mask_layer4_5_6",
        "label":       "Group 3: mask layer 4+5+6 (>14A)",
        "mask_layers": [4, 5, 6],
        "fixed_mode":  "core_shell",
    },
    {
        "name":        "mask_layer3_6",
        "label":       "Group 4: mask layer 3+4+5+6 (shell fixed)",
        "mask_layers": [3, 4, 5, 6],
        "fixed_mode":  "core_shell",
    },
    {
        "name":        "mask_layer2_6",
        "label":       "Group 5: mask layer 2+3+4+5+6 (core only fixed)",
        "mask_layers": [2, 3, 4, 5, 6],
        "fixed_mode":  "core_only",
    },
]

# ============================================================
# 3. Protein definitions
# ============================================================
PROTEINS = [
    # ---- CA II (human, PDB: 1CA2) ----
    # PDB numbering starts at 4 (residues 1-3 missing), gap at 126
    # => PDB resnum ≠ sequence index; mapping verified per-residue
    {
        "name":       "CA2",
        "pdb_path":   "1CA2.pdb",
        "pdb_chain":  "A",
        "output_file": "results_CA2.txt",
        "structure_dir": "pdbs_CA2",
        "seq_length": 256,
        # Layer 1: Catalytic core (5 residues)
        # PDB: His94 (Zn), His96 (Zn), Glu106, His119 (Zn), Thr199
        "layer_1": {91, 93, 103, 116, 195},
        # Layer 2: Functional shell (11 residues)
        # PDB: Asn62, His64, Asn67, Gln92, Val121, Phe131, Val143, Leu198, Thr200, Pro202, Trp209
        "layer_2": {59, 61, 64, 89, 118, 127, 139, 194, 196, 198, 205},
        # Layer 3: Inner zone, <14A from Zn (54 residues)
        "layer_3": {
            3, 4, 10, 25, 26, 27, 28, 58, 60, 62, 63, 65, 66, 87, 88,
            90, 92, 94, 95, 101, 102, 104, 105, 112, 113, 114, 115, 117, 119, 137,
            138, 140, 141, 142, 143, 144, 192, 193, 197, 199, 200, 201, 202, 203, 204,
            206, 207, 238, 239, 240, 241, 242, 243, 244
        },
        # Layer 4: Middle zone, 14-17A from Zn (42 residues)
        "layer_4": {
            2, 9, 13, 20, 24, 29, 30, 56, 57, 67, 85, 86, 96, 100, 106,
            110, 111, 120, 128, 130, 131, 136, 145, 156, 166, 177, 180, 181, 188, 189,
            190, 191, 208, 211, 212, 219, 222, 223, 225, 227, 237, 245
        },
        # Layer 5: Outer zone, 17-20A from Zn (81 residues)
        "layer_5": {
            1, 5, 6, 8, 11, 12, 14, 16, 17, 18, 19, 21, 22, 23, 44,
            46, 48, 51, 53, 54, 55, 68, 69, 74, 75, 76, 77, 80, 81, 97,
            98, 99, 107, 108, 109, 121, 126, 129, 132, 133, 134, 135, 146, 150, 152,
            153, 154, 155, 157, 159, 160, 163, 164, 165, 167, 168, 169, 172, 175, 176,
            178, 179, 182, 185, 186, 187, 209, 210, 213, 214, 218, 220, 221, 224, 226,
            228, 235, 236, 246, 247, 252
        },
        # Layer 6: Distal region, >20A from Zn (63 residues)
        "layer_6": {
            7, 15, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            45, 47, 49, 50, 52, 70, 71, 72, 73, 78, 79, 82, 83, 84, 122,
            123, 124, 125, 147, 148, 149, 151, 158, 161, 162, 170, 171, 173, 174, 183,
            184, 215, 216, 217, 229, 230, 231, 232, 233, 234, 248, 249, 250, 251, 253,
            254, 255, 256
        },
    },

    # ---- CA IX (human, PDB: 3IAI chain A) ----
    # PDB has insertion codes (0A, 0B, 50A, 54A, 54B, etc.) and gaps
    # => PDB resnum ≠ sequence index; mapping verified per-residue
    {
        "name":       "CA9",
        "pdb_path":   "3IAI_A.pdb",
        "pdb_chain":  "A",
        "output_file": "results_CA9.txt",
        "structure_dir": "pdbs_CA9",
        "seq_length": 257,
        # Layer 1: Catalytic core (5 residues)
        # PDB: His94 (Zn), His96 (Zn), Glu106, His119 (Zn), Thr199
        "layer_1": {92, 94, 104, 117, 198},
        # Layer 2: Functional shell (11 residues)
        # PDB: Asn62, His64, Gln67, Gln92, Val121, Val131, Val143, Leu198, Thr200, Pro202, Trp209
        "layer_2": {64, 66, 69, 90, 119, 128, 140, 197, 199, 201, 208},
        # Layer 3: Inner zone, <14A from Zn (51 residues)
        "layer_3": {
            8, 9, 27, 28, 29, 30, 63, 65, 67, 68, 70, 71, 88, 89, 91,
            93, 95, 96, 102, 103, 105, 106, 114, 115, 116, 118, 120, 138, 139, 141,
            142, 143, 144, 145, 195, 196, 200, 202, 203, 204, 206, 207, 209, 210, 238,
            239, 240, 241, 242, 243, 244
        },
        # Layer 4: Middle zone, 14-17A from Zn (49 residues)
        "layer_4": {
            5, 6, 7, 10, 15, 19, 21, 22, 26, 31, 32, 61, 62, 72, 87,
            97, 101, 107, 111, 112, 113, 121, 131, 132, 133, 136, 137, 146, 155, 158,
            179, 182, 183, 184, 191, 192, 193, 194, 205, 211, 214, 215, 222, 225, 226,
            229, 231, 237, 245
        },
        # Layer 5: Outer zone, 17-20A from Zn (68 residues)
        "layer_5": {
            4, 11, 12, 13, 14, 16, 18, 20, 23, 24, 25, 46, 48, 51, 59,
            60, 73, 74, 76, 77, 78, 79, 80, 86, 100, 109, 110, 122, 127, 129,
            130, 134, 135, 147, 154, 156, 157, 159, 161, 162, 165, 166, 167, 168, 169,
            170, 171, 178, 180, 181, 187, 190, 212, 213, 216, 217, 221, 223, 224, 227,
            228, 230, 232, 235, 236, 246, 247, 252
        },
        # Layer 6: Distal region, >20A from Zn (73 residues)
        "layer_6": {
            1, 2, 3, 17, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 75, 81, 82,
            83, 84, 85, 98, 99, 108, 123, 124, 125, 126, 148, 149, 150, 151, 152,
            153, 160, 163, 164, 172, 173, 174, 175, 176, 177, 185, 186, 188, 189, 218,
            219, 220, 233, 234, 248, 249, 250, 251, 253, 254, 255, 256, 257
        },
    },

    # ---- CPA (bovine, PDB: 5CPA) ----
    # PDB numbering 1-307 is sequential (no gaps, no insertion codes)
    # => PDB resnum = sequence index
    {
        "name":       "CPA",
        "pdb_path":   "5CPA.pdb",
        "pdb_chain":  "A",
        "output_file": "results_CPA.txt",
        "structure_dir": "pdbs_CPA",
        "seq_length": 307,
        # Layer 1: Catalytic core (5 residues)
        # His69 (Zn), Glu72 (Zn), Arg127 , His196 (Zn), Glu270
        "layer_1": {69, 72, 127, 196, 270},
        # Layer 2: Functional shell (7 residues)
        # Arg71, Asn144, Arg145, Ser197, Tyr248, Thr268, Phe279
        "layer_2": {71, 144, 145, 197, 248, 268, 279},
        # Layer 3: Inner zone, <14A from Zn (59 residues)
        "layer_3": {
            64, 65, 66, 67, 68, 70, 73, 74, 75, 76, 78, 109, 110, 111, 112,
            115, 116, 119, 125, 126, 128, 129, 141, 142, 143, 146, 156, 163, 164, 165,
            166, 193, 194, 195, 198, 199, 200, 201, 202, 203, 207, 243, 247, 250, 251,
            252, 253, 254, 255, 267, 269, 271, 272, 273, 274, 275, 278, 280, 281
        },
        # Layer 4: Middle zone, 14-17A from Zn (46 residues)
        "layer_4": {
            63, 77, 79, 107, 108, 113, 114, 117, 118, 123, 124, 130, 140, 147, 149,
            155, 157, 162, 167, 173, 175, 176, 179, 192, 204, 205, 206, 208, 238, 239,
            240, 241, 242, 244, 246, 249, 256, 257, 258, 266, 276, 277, 286, 289, 290,
            293
        },
        # Layer 5: Outer zone, 17-20A from Zn (61 residues)
        "layer_5": {
            12, 13, 14, 15, 18, 41, 42, 47, 48, 49, 62, 80, 81, 82, 106,
            120, 121, 122, 131, 138, 139, 148, 150, 151, 152, 153, 154, 158, 159, 160,
            161, 168, 169, 170, 171, 172, 174, 177, 178, 180, 191, 209, 223, 226, 227,
            230, 236, 237, 245, 259, 265, 282, 283, 285, 287, 288, 291, 292, 294, 296,
            297
        },
        # Layer 6: Distal region, >20A from Zn (129 residues)
        "layer_6": {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 83, 84, 85, 86, 87, 88, 89, 90, 91,
            92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 132,
            133, 134, 135, 136, 137, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
            210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225,
            228, 229, 231, 232, 233, 234, 235, 260, 261, 262, 263, 264, 284, 295, 298,
            299, 300, 301, 302, 303, 304, 305, 306, 307
        },
    },
]

# ============================================================
# 4. Validation
# ============================================================
for _p in PROTEINS:
    _total = sum(len(_p[f"layer_{i}"]) for i in range(1, 7))
    assert _total == _p["seq_length"], (
        f"{_p['name']}: layer total {_total} != seq_length {_p['seq_length']}"
    )