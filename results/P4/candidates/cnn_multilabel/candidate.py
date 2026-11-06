import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
_D = Path(__file__).parent
sys.path.insert(0, str(_D))
from model_def import ECGNet  # definizione della rete accanto a questo file
_CFG = __import__("json").loads((_D / "config.json").read_text())
_NET = None
def predict(df, lib):
    global _NET
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _NET is None:
        _NET = ECGNet(_CFG["n_out"], _CFG["n_tab"], False).to(dev)
        _NET.load_state_dict(torch.load(_D / "weights.pt", map_location=dev)); _NET.eval()
    data = lib.load_data(allow_test=os.environ.get("ECHONEXT_ALLOW_TEST") == "1")
    out = np.zeros(len(df), dtype=np.float32)
    for split in df["split"].unique():
        idx = np.where((df["split"] == split).values)[0]
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            x = torch.from_numpy(np.ascontiguousarray(data.waveforms(split, df.row_in_split.values[b]).transpose(0, 2, 1), dtype=np.float32)).to(dev)
            t = torch.zeros(len(b), _CFG["n_tab"], device=dev)
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                out[b] = torch.sigmoid(_NET(x, t).float()[:, 0]).cpu().numpy()
    return pd.DataFrame({"score": out, "in_scope": True}, index=df.index)
