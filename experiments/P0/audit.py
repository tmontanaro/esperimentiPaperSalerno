"""P0 data audit. Uso: python3 audit.py [debug_n]. Solo aggregati; test mai letto (solo header .npy e conteggi)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import numpy as np, pandas as pd
import echonext_lib as lib

debug_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
d = lib.load_data(debug_n=debug_n)
out = {"debug_n": debug_n, "data_dir": str(d.data_dir), "meta_file": d.meta_path.name}
integ = d.integrity
out["integrity"] = {k: v for k, v in integ.items() if k != "columns"}
overlap = sum(integ["patient_overlap"].values())
expected = lib.ECG_FEATURES + lib.FLAG_COLS + ["lvef_value", "qrs_duration", "patient_key", "split"]
missing_cols = [c for c in expected if c not in integ["columns"]]
out["missing_expected_columns"] = missing_cols
print("Split counts:", integ["split_counts"], "| overlap:", integ["patient_overlap"], "| max ECG/paz:", integ["ecg_per_patient_max"])
print("Colonne attese mancanti:", missing_cols)

# mancanti
cols = [c for c in expected if c in integ["columns"] and c not in ("patient_key", "split")]
miss = pd.DataFrame({s: df[cols].isna().mean() for s, df in (("train", d.train), ("val", d.val))})
lib.show(miss)
out["missing_frac"] = {s: miss[s].round(5).to_dict() for s in miss}

# prevalenze + etichette granulari mancanti
prev = {}
rows = []
for s, df in (("train", d.train), ("val", d.val)):
    for f in lib.FLAG_COLS:
        v = df[f].dropna()
        rows.append({"split": s, "flag": f, "n_labeled": len(v), "n_pos": int(v.sum()), "prev": v.mean(), "frac_nan": df[f].isna().mean()})
pv = pd.DataFrame(rows)
lib.show(pv.pivot(index="flag", columns="split", values=["prev", "n_pos", "frac_nan"]))
for s in ("train", "val"):
    prev[s] = {r.flag: {"n_labeled": int(r.n_labeled), "n_pos": int(r.n_pos), "prev": round(float(r.prev), 5), "frac_nan": round(float(r.frac_nan), 5)}
               for r in pv[pv.split == s].itertuples()}
out["label_profile"] = prev
gran = [f for f in lib.FLAG_COLS if f != lib.TARGET]
out["granular_missing_pattern"] = {}
for s, df in (("train", d.train), ("val", d.val)):
    out["granular_missing_pattern"][s] = {
        "target_nan": float(df[lib.TARGET].isna().mean()),
        "all_granular_nan": float(df[gran].isna().all(axis=1).mean()),
        "any_granular_nan": float(df[gran].isna().any(axis=1).mean()),
        "lvef_value_nan": float(df["lvef_value"].isna().mean())}
print(json.dumps(out["granular_missing_pattern"]))

# distribuzioni
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
dist = {}
for s, df in (("train", d.train), ("val", d.val)):
    for c in ("qrs_duration", "age_at_ecg"):
        dist[f"{s}_{c}"] = df[c].quantile(q).round(2).to_dict()
lib.show(pd.DataFrame(dist).T)
out["quantiles"] = {k: {str(a): b for a, b in v.items()} for k, v in dist.items()}
out["lvef_quantiles_train"] = {str(a): round(float(b), 2) for a, b in d.train["lvef_value"].quantile(q).items()}
out["sex_male_frac"] = {"train": float(d.train.sex.mean()), "val": float(d.val.sex.mean())}
corr = d.train[lib.ECG_FEATURES].corr()
lib.show(corr)
out["corr_features_train"] = corr.round(3).to_dict()

# forme d'onda (solo header)
wf = {}
ok_wave = True
for s in ("train", "val", "test"):
    p = d.wave_path(s)
    n_csv = int(integ["split_counts"].get(s, 0))
    if not p.exists():
        wf[s] = {"exists": False, "n_csv": n_csv}; ok_wave = False; continue
    sh = tuple(int(x) for x in np.load(p, mmap_mode="r").shape)
    good = len(sh) == 4 and sh[1:] == (1, 2500, 12)
    wf[s] = {"exists": True, "shape": list(sh), "n_csv": n_csv, "n_match": sh[0] == n_csv, "shape_ok": good}
    ok_wave &= good and sh[0] == n_csv
out["waveforms"] = wf
print(wf)

try:
    import torch
    gpu = bool(torch.cuda.is_available()); out["torch"] = torch.__version__
except Exception as e:
    gpu = False; out["torch"] = f"assente ({e})"
out["gpu"] = gpu
out["integrity_ok"] = bool(overlap == 0 and not missing_cols)
out["waveforms_ok"] = bool(ok_wave)
out["n_train"], out["n_val"] = int(integ["split_counts"].get("train", 0)), int(integ["split_counts"].get("val", 0))
out["prev_target_train"] = prev["train"][lib.TARGET]["prev"]
if debug_n:
    out["n_train_loaded"], out["n_val_loaded"] = len(d.train), len(d.val)
dest = ROOT / "results" / "P0" / ("debug_profile.json" if debug_n else "data_profile.json")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, default=str))
print("scritto", dest)
