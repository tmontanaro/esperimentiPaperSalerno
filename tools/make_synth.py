"""Genera un dataset sintetico con lo stesso schema di EchoNext (solo per testare i notebook)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
rng = np.random.default_rng(0)

n_pat = int(N * 0.8)
patient = rng.integers(0, n_pat, N)
# split per paziente (nessuna sovrapposizione), val/test = ultimo ECG del paziente
pat_split = rng.choice(["train", "val", "test"], n_pat, p=[0.7, 0.15, 0.15])
split = pat_split[patient].astype(object)
df = pd.DataFrame({"patient_key": patient, "split": split})
df["acquisition_year"] = rng.integers(2008, 2023, N)
df["location_setting"] = rng.choice(["inpatient", "emergency", "outpatient", "procedural"], N)
df["race_ethnicity"] = rng.choice(["hispanic", "white", "black", "asian", "other", "unknown"], N)
df["age_at_ecg"] = np.clip(rng.normal(60, 17, N), 18, 90).round()
df["sex"] = rng.choice(["male", "female"], N)
df["ventricular_rate"] = rng.normal(78, 18, N).round()
df["atrial_rate"] = df["ventricular_rate"] + rng.normal(0, 3, N)
df["pr_interval"] = rng.normal(165, 28, N).round()
df["qrs_duration"] = np.clip(rng.gamma(20, 4.6, N), 70, 220).round()
df["qt_corrected"] = rng.normal(440, 35, N).round()
miss = rng.random(N) < 0.08
df.loc[miss, ["pr_interval", "atrial_rate"]] = np.nan

wide = (df.qrs_duration > 120).astype(float)
male = (df.sex == "male").astype(float)
z = -1.6 + 0.02 * (df.age_at_ecg - 60) + 1.2 * wide + 0.4 * miss + 0.015 * (df.ventricular_rate - 78)
lvef = np.clip(58 - 0.1 * (df.age_at_ecg - 60) - 14 * wide * rng.random(N) - 6 * male + rng.normal(0, 7, N), 10, 75)
df["lvef_value"] = lvef.round()
p_shd = 1 / (1 + np.exp(-z)); shd = rng.random(N) < p_shd
df["lvef_lte_45_flag"] = (lvef <= 45).astype(int)
df["lvwt_gte_13_flag"] = (rng.random(N) < 0.12 + 0.05 * (df.age_at_ecg > 70)).astype(int)
for c in ["aortic_stenosis", "aortic_regurgitation", "mitral_regurgitation", "tricuspid_regurgitation"]:
    df[f"{c}_moderate_or_greater_flag"] = (rng.random(N) < 0.05 + 0.1 * (df.age_at_ecg > 75)).astype(int)
df["pulmonary_regurgitation_moderate_or_greater_flag"] = (rng.random(N) < 0.01).astype(int)
df["rv_systolic_dysfunction_moderate_or_greater_flag"] = (rng.random(N) < 0.06 + 0.1 * (lvef < 40)).astype(int)
df["pericardial_effusion_moderate_large_flag"] = (rng.random(N) < 0.02).astype(int)
df["pasp_gte_45_flag"] = (rng.random(N) < 0.1).astype(int)
df["tr_max_gte_32_flag"] = (rng.random(N) < 0.1).astype(int)
flags = [c for c in df.columns if c.endswith("_flag")]
df["shd_moderate_or_greater_flag"] = (df[flags].sum(axis=1) > 0).astype(int) | shd.astype(int)
# ECG senza eco entro 1 anno: etichette granulari mancanti (come nel dataset reale)
nogran = rng.random(N) < 0.2
for c in flags + ["lvef_value"]:
    if c != "shd_moderate_or_greater_flag":
        df.loc[nogran, c] = np.nan
# echo-derived numeric/categorical columns (NON da usare come input)
df["ivs_measurement"] = rng.normal(1.0, 0.2, N)
df["pasp_value"] = rng.normal(32, 10, N)
df["most_recent_ecg"] = 0
last = df.groupby("patient_key").cumcount(ascending=False) == 0
df.loc[last, "most_recent_ecg"] = 1
# val/test: solo ultimo ECG per paziente; gli altri -> no_split
df.loc[(df.split.isin(["val", "test"])) & (df.most_recent_ecg == 0), "split"] = "no_split"
df.to_csv(out / "EchoNext_metadata_100k.csv", index=False)

# forme d'onda sintetiche: N_split x 1 x 2500 x 12
t = np.arange(2500) / 250.0
for s in ["train", "val", "test", "no_split"]:
    sub = df[df.split == s]
    n = len(sub)
    W = np.zeros((n, 1, 2500, 12), dtype=np.float32)
    hr = (sub.ventricular_rate.values / 60.0)
    for i in range(n):
        base = np.sin(2 * np.pi * hr[i] * t) ** 20  # picchi R finti
        width = 1.0 + (sub.qrs_duration.values[i] - 90) / 100
        for l in range(12):
            W[i, 0, :, l] = base * (1 + 0.1 * l) / width + 0.05 * rng.standard_normal(2500)
    np.save(out / f"EchoNext_{s}_waveforms.npy", W)
    print(s, W.shape)
print(df.split.value_counts().to_dict(), "prev SHD", df.shd_moderate_or_greater_flag.mean().round(3))
