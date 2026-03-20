from structure import load_pdb, build_prompt

wt = load_pdb()

# 正常实验组
prompt_normal = build_prompt(wt, mask_layers=[3, 4], no_coords=False)
print(f"正常实验组 coordinates is None: {prompt_normal.coordinates is None}")
print(f"正常实验组 坐标有多少nan: {prompt_normal.coordinates.isnan().sum().item()}")

# 对照组
prompt_control = build_prompt(wt, mask_layers=[3, 4], no_coords=True)
print(f"对照组 coordinates is None: {prompt_control.coordinates is None}")