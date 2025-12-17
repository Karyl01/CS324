import torch
import numpy as np

bundle = torch.load("saved_model/best_mlp.pt", map_location="cpu", weights_only=False)
scaler = bundle["scaler"]

# 把 scaler 变成纯 numpy 参数（不再需要 sklearn）
bundle["scaler_mean"] = scaler.mean_.astype(np.float32)
bundle["scaler_scale"] = scaler.scale_.astype(np.float32)
bundle.pop("scaler", None)

torch.save(bundle, "saved_model/best_mlp_nosklearn.pt")
print("saved -> saved_model/best_mlp_nosklearn.pt")
