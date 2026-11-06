"""P0 data audit. Solo aggregati; test: solo conteggi/id e header .npy.
Uso: python3 experiments/P0/audit.py [debug_n]   (con debug_n non scrive il profilo)"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while not (ROOT / "lib").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "lib"))
import numpy as np, pandas as pd
import echonext_lib as lib

debug_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
d = lib.load_data(debug_n=debug_n)
prof = {"debug_n": debug_n}
integ = d.integrity
prof["integrity"] = {k: v for k, v in integ.items() if k != "columns"}
cols = integ["columns"]
need = lib.ECG_FEATURES + lib.FLAG_COLS + ["lvef_value", "qrs_duration", "patient_key", "split"]
missing_cols = sorted(set(need) - set(cols))
prof["missing_columns"] = missing_cols
print("Colonne attese mancanti:", missing_cols)
print("Integrità:", prof["integrity"])

# mancanti
miss = pd.DataFrame({s: df[[c for c in need if c in df.columns]].isna().mean() for s, df in (("train", d.train), ("val", d.val))})
prof["missing_frac"] = miss.round(5).to_dict()
lib.show(miss)

# etichette
rows = []
for c in lib.FLAG_COLS:
    r = {"label": c}
    for s, df in (("train", d.train), ("val", d.val)):
        v = df[c]
        r[f"{s}_n_labeled"] = int(v.notna().sum()); r[f"{s}_frac_missing"] = float(v.isna().mean())
        r[f"{s}_n_pos"] = int((v == 1).sum()); r[f"{s}_prev"] = float(v.dropna().mean())
    rows.append(r)
lab = pd.DataFrame(rows).set_index("label")
prof["labels"] = lab.round(5).to_dict("index")
lib.show(lab)

# QRS / età
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
dist = pd.DataFrame({f"{c}_{s}": df[c].quantile(q) for c in ("qrs_duration", "age_at_ecg") for s, df in (("train", d.train), ("val", d.val))})
prof["quantiles"] = dist.round(2).to_dict()
lib.show(dist)
corr = d.train[lib.ECG_FEATURES].corr(method="spearman")
prof["spearman_features_train"] = corr.round(3).to_dict()
lib.show(corr)

# forme d'onda (solo header)
wf = {}
full = pd.read_csv(d.meta_path, usecols=["split"]).split.value_counts().to_dict()
for s in ("train", "val", "test"):
    p = d.wave_path(s)
    e = {"exists": p.exists(), "csv_rows": int(full.get(s, 0))}
    if p.exists():
        # header letto direttamente (non richiede che il file sia completo)
        with open(p, "rb") as f:
            ver = np.lib.format.read_magic(f)
            rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
            shp, _, dt = rd(f)
            off = f.tell()
        sh = [int(x) for x in shp]
        exp_bytes = off + int(np.prod(sh)) * dt.itemsize
        real = p.stat().st_size
        row_bytes = int(np.prod(sh[1:])) * dt.itemsize
        e.update(shape=sh, dtype=str(dt), shape_ok=(len(sh) == 4 and sh[1:] == [1, 2500, 12]),
                 n_match=(sh[0] == e["csv_rows"]), file_bytes=real, expected_bytes=exp_bytes,
                 file_complete=(real >= exp_bytes), complete_rows_in_file=int((real - off) // row_bytes))
        try:
            np.load(p, mmap_mode="r")
            e["mmap_ok"] = True
        except Exception as ex:
            e["mmap_ok"] = False; e["mmap_error"] = str(ex)[:80]
    wf[s] = e
prof["waveforms"] = wf
lib.show(pd.DataFrame(wf).T.astype(str))

# GPU
try:
    import torch
    prof["gpu"] = bool(torch.cuda.is_available()); prof["torch"] = torch.__version__
except Exception as ex:
    prof["gpu"] = False; prof["torch"] = None; prof["torch_error"] = str(ex)[:100]
print("GPU:", prof["gpu"])

ov = sum(prof["integrity"]["patient_overlap"].values())
wf_ok = all(w["exists"] and w.get("shape_ok") and w.get("n_match") and w.get("file_complete") and w.get("mmap_ok") for w in wf.values())
prof["integrita"] = "OK" if (ov == 0 and not missing_cols) else "PROBLEMI"
prof["waveforms_status"] = "presenti" if wf_ok else "mancanti"
prof["n_train"] = int(full.get("train", 0)); prof["n_val"] = int(full.get("val", 0))
prof["prev_target_train"] = prof["labels"][lib.TARGET]["train_prev"]
if debug_n is None:
    (ROOT / "results/P0").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/P0/data_profile.json").write_text(json.dumps(prof, indent=1, default=str))
print("ESITO:", prof["integrita"], prof["waveforms_status"])
