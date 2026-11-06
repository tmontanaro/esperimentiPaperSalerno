import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import numpy as np, pandas as pd
import echonext_lib as lib

debug = int(sys.argv[1]) if len(sys.argv) > 1 else None
d = lib.load_data(debug_n=debug)
ig = d.integrity
prof = {"debug_n": debug, "integrity": {k: v for k, v in ig.items() if k != "columns"}}
print("== integrità"); print(json.dumps(prof["integrity"], indent=1, default=str))
cols = set(ig["columns"])
need = lib.ECG_FEATURES + lib.FLAG_COLS + ["lvef_value", "qrs_duration", "patient_key", "split"]
missing_cols = sorted(set(need) - cols)
prof["missing_columns"] = missing_cols
print("colonne attese mancanti:", missing_cols)
tv = pd.concat([d.train, d.val])
avail = [c for c in need if c in tv.columns and c not in ("split", "patient_key")]
miss = pd.DataFrame({s: df[avail].isna().mean() for s, df in (("train", d.train), ("val", d.val))})
print("== frazione mancanti"); lib.show(miss)
prof["missing_frac"] = {s: miss[s].round(5).to_dict() for s in miss.columns}
prev = {}
for s, df in (("train", d.train), ("val", d.val)):
    rows = {}
    for c in lib.FLAG_COLS:
        if c in df:
            rows[c] = {"n_labeled": int(df[c].notna().sum()), "frac_missing": float(df[c].isna().mean()),
                       "n_pos": int(df[c].sum()), "prev": float(df[c].mean())}
    prev[s] = rows
    print("== prevalenza", s); lib.show(pd.DataFrame(rows).T)
prof["prevalence"] = prev
q = [.01, .05, .25, .5, .75, .95, .99]
dist = {}
for c in ("qrs_duration", "age_at_ecg"):
    dist[c] = {s: df[c].quantile(q).round(2).to_dict() for s, df in (("train", d.train), ("val", d.val))}
    print("==", c); lib.show(pd.DataFrame(dist[c]))
dist["qrs_ge_120_frac"] = {s: float((df.qrs_duration >= 120).mean()) for s, df in (("train", d.train), ("val", d.val))}
dist["sex_male_frac"] = {s: float(df.sex.mean()) for s, df in (("train", d.train), ("val", d.val))}
print(dist["qrs_ge_120_frac"], dist["sex_male_frac"])
prof["distributions"] = dist
corr = d.train[lib.ECG_FEATURES].corr().round(3)
print("== correlazioni (train)"); lib.show(corr); prof["feature_corr_train"] = corr.to_dict()
wf = {}
counts = ig["split_counts"]
for s in ("train", "val", "test"):
    p = d.wave_path(s)
    e = {"exists": p.exists(), "n_csv": int(counts.get(s, 0))}
    if p.exists():
        sh = tuple(np.load(p, mmap_mode="r").shape)  # solo header
        e["shape"] = list(sh)
        e["shape_ok"] = bool(len(sh) == 4 and sh[1:] == (1, 2500, 12))
        e["n_matches_csv"] = bool(sh[0] == e["n_csv"])
    wf[s] = e
print(wf); prof["waveforms"] = wf
try:
    import torch
    gpu = bool(torch.cuda.is_available()); prof["torch"] = torch.__version__
except Exception as ex:
    gpu = False; prof["torch"] = None; print("torch mancante", ex)
prof["gpu"] = gpu; print("GPU", gpu)
ov = ig["patient_overlap"]
prof["checks"] = {"no_overlap": all(v == 0 for v in ov.values()),
                  "one_ecg_val_test": ig["ecg_per_patient_max"]["val"] == 1 and ig["ecg_per_patient_max"]["test"] == 1,
                  "columns_ok": not missing_cols,
                  "waveforms_present": all(w["exists"] for w in wf.values()),
                  "waveforms_shape_ok": all(w.get("shape_ok", False) for w in wf.values()),
                  "waveforms_n_match": all(w.get("n_matches_csv", False) for w in wf.values())}
prof["n_train"] = int(counts.get("train", 0)); prof["n_val"] = int(counts.get("val", 0)); prof["n_test"] = int(counts.get("test", 0))
print(prof["checks"])
if not debug:
    (ROOT / "results/P0").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/P0/data_profile.json").write_text(json.dumps(prof, indent=1, default=str))
