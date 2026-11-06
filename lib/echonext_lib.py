"""Libreria condivisa da tutti gli agenti EchoNext.

Garanzie imposte dal codice (non solo dalle istruzioni):
  * il TEST SET non viene mai caricato, a meno che `tools/final_eval.py` imposti ECHONEXT_ALLOW_TEST=1;
  * le righe 'test' e 'no_split' vengono scartate subito dopo la lettura del CSV;
  * le feature in ingresso ai modelli sono solo quelle ECG (nessuna colonna derivata dall'eco);
  * `show()` rifiuta di stampare tabelle a livello di paziente (solo aggregati).
"""
from __future__ import annotations

import json
import os
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SEED = 42

TARGET = "shd_moderate_or_greater_flag"
ECG_FEATURES = ["sex", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected", "age_at_ecg"]
FLAG_COLS = [
    "shd_moderate_or_greater_flag", "lvef_lte_45_flag", "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag", "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag", "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag", "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag", "pasp_gte_45_flag", "tr_max_gte_32_flag",
]
# Colonne derivate dall'ecocardiogramma: MAI come input (leakage)
ECHO_ONLY_COLS = [
    "aortic_stenosis_value", "aortic_regurgitation_value", "mitral_regurgitation_value",
    "tricuspid_regurgitation_value", "pulmonary_regurgitation_value", "rv_systolic_function_value",
    "pericardial_effusion_value", "ivs_measurement", "lvpw_measurement", "pasp_value",
    "tr_max_velocity_value", "lvef_value",
] + FLAG_COLS
assert not set(ECG_FEATURES) & set(ECHO_ONLY_COLS)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# --------------------------------------------------------------------------- dati
def locate_dataset(root) -> Path:
    """Restituisce il percorso del CSV dei metadati (anche in sottocartelle o dentro uno zip, che viene estratto)."""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Cartella non trovata: {root}")

    def find_csv(base: Path):
        c = [p for p in base.rglob("*") if p.is_file() and "metadata" in p.name.lower()
             and p.name.lower().endswith((".csv", ".csv.gz"))]
        return sorted(c, key=lambda p: ("100k" not in p.name.lower(), len(str(p))))

    hits = find_csv(root)
    if hits:
        return hits[0]
    target = root / "extracted"
    for z in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            if any("metadata" in n.lower() and ".csv" in n.lower() for n in zf.namelist()):
                zf.extractall(target)
                break
    hits = find_csv(target) if target.exists() else []
    if hits:
        return hits[0]
    files = sorted(p for p in root.rglob("*") if p.is_file())
    listing = "; ".join(f"{p.relative_to(root)} ({p.stat().st_size / 1e6:.2f} MB)" for p in files[:15])
    raise FileNotFoundError(f"Nessun CSV di metadata in {root}. Trovati {len(files)} file: {listing or 'nessuno'}")


def _configured_data_dir() -> str | None:
    if os.environ.get("ECHONEXT_DIR"):
        return os.environ["ECHONEXT_DIR"]
    st = ROOT / "agentic" / "pipeline_state.json"
    if st.exists():
        return json.loads(st.read_text()).get("config", {}).get("data_dir")
    return None


@dataclass
class Data:
    data_dir: Path
    meta_path: Path
    train: pd.DataFrame
    val: pd.DataFrame
    integrity: dict
    _test: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def test(self) -> pd.DataFrame:
        if self._test is None:
            raise PermissionError("Test set bloccato: lo può aprire solo tools/final_eval.py (fase P7).")
        return self._test

    def wave_path(self, split: str) -> Path:
        return self.data_dir / f"EchoNext_{split}_waveforms.npy"

    def waveforms(self, split: str, rows) -> np.ndarray:
        """Forme d'onda (n, 2500, 12) per le righe `row_in_split` richieste (lettura mmap)."""
        if split == "test" and self._test is None:
            raise PermissionError("Forme d'onda di test bloccate.")
        W = np.load(self.wave_path(split), mmap_mode="r")
        return np.asarray(W[np.asarray(rows), 0])


def load_data(data_dir=None, debug_n: int | None = None, allow_test: bool = False) -> Data:
    data_dir = data_dir or _configured_data_dir()
    if not data_dir:
        raise ValueError("Imposta ECHONEXT_DIR oppure config.data_dir in agentic/pipeline_state.json")
    if allow_test and os.environ.get("ECHONEXT_ALLOW_TEST") != "1":
        raise PermissionError("allow_test=True è consentito solo a tools/final_eval.py")
    meta_path = locate_dataset(data_dir)
    meta = pd.read_csv(meta_path)
    meta["sex"] = meta["sex"].astype(str).str.lower().map(
        {"male": 1, "female": 0, "m": 1, "f": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    if meta["sex"].notna().mean() < 0.9:
        raise ValueError("Colonna 'sex' non interpretabile")
    # l'ordine delle righe del CSV coincide con l'ordine dei file .npy di ciascuno split
    meta["row_in_split"] = meta.groupby("split").cumcount()

    # controlli di integrità che coinvolgono 'test' usando SOLO conteggi e id (nessuna etichetta)
    sets = {s: set(meta.loc[meta.split == s, "patient_key"]) for s in ("train", "val", "test")}
    integrity = {
        "split_counts": meta.split.value_counts().to_dict(),
        "patient_overlap": {"train_val": len(sets["train"] & sets["val"]), "train_test": len(sets["train"] & sets["test"]),
                            "val_test": len(sets["val"] & sets["test"])},
        "ecg_per_patient_max": {s: int(meta[meta.split == s].groupby("patient_key").size().max()) for s in ("train", "val", "test")},
        "columns": list(meta.columns),
    }
    test = meta[meta.split == "test"].copy() if allow_test else None
    meta = meta[meta.split.isin(["train", "val"])]
    if debug_n:
        meta = meta.groupby("split").head(debug_n)
        if test is not None:
            test = test.head(debug_n)
    return Data(Path(meta_path).parent, meta_path, meta[meta.split == "train"].copy(),
                meta[meta.split == "val"].copy(), integrity, test)


# --------------------------------------------------------------------------- output sicuro
def show(df, max_rows: int = 60):
    """Stampa solo tabelle aggregate: rifiuta frame a livello di paziente o troppo lunghi."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if "patient_key" in df.columns or "row_in_split" in df.columns:
        raise ValueError("show(): la tabella contiene identificativi di paziente; stampa solo aggregati.")
    if len(df) > max_rows:
        raise ValueError(f"show(): {len(df)} righe > {max_rows}; aggrega prima di stampare.")
    print(df.round(4).to_string())


# --------------------------------------------------------------------------- metriche
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score  # noqa: E402


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def boot_ci(y, p, fn=safe_auc, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], p[i])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def metrics_binary(y, p, n_boot: int = 300) -> dict:
    y = np.asarray(y)
    lo, hi = boot_ci(y, p, n=n_boot)
    return {"n": int(len(y)), "n_pos": int(y.sum()), "prev": float(y.mean()) if len(y) else float("nan"),
            "auroc": safe_auc(y, p), "auroc_lo": lo, "auroc_hi": hi,
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")}


def metrics_regression(y, p) -> dict:
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))}


# --------------------------------------------------------------------------- dataset tabellare e modelli
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def get_xy(df: pd.DataFrame, target: str, features=ECG_FEATURES):
    """X, y e il frame filtrato (righe con target mancante escluse). Rifiuta feature derivate dall'eco."""
    bad = set(features) & set(ECHO_ONLY_COLS)
    if bad:
        raise ValueError(f"Leakage: feature derivate dall'eco non ammesse come input: {sorted(bad)}")
    d = df.dropna(subset=[target])
    y = d[target].values
    y = y.astype(int) if target in FLAG_COLS else y.astype(float)
    return d[list(features)].astype(float), y, d


def model_zoo(seed: int = SEED) -> dict:
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)  # noqa: E731
    return {
        "logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "lasso_logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.1)),
        "random_forest": make_pipeline(imp(), RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=seed)),
        "hist_gb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=seed),
        "mlp_dense": make_pipeline(imp(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                          early_stopping=True, max_iter=200, random_state=seed)),
    }


def eval_models(Xtr, ytr, Xva, yva, models: dict | None = None):
    """Addestra su train e valuta su validation. Ritorna (tabella, predizioni)."""
    rows, preds = [], {}
    for name, m in (models or model_zoo()).items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        rows.append({"model": name, **metrics_binary(yva, p), "sec": round(time.time() - t0, 1)})
        preds[name] = p
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True), preds


def oof_proba(model, X, y, groups, n_splits: int = 5):
    """Predizioni out-of-fold con CV raggruppata per paziente."""
    return cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits), method="predict_proba")[:, 1]


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["qrs_bin"] = pd.cut(d.qrs_duration, [0, 100, 120, 10000], labels=["<100", "100-119", ">=120"], right=False).astype(object)
    d["age_bin"] = pd.cut(d.age_at_ecg, [0, 50, 65, 75, 200], labels=["<50", "50-64", "65-74", ">=75"], right=False).astype(object)
    return d


# --------------------------------------------------------------------------- candidati (contratto per la valutazione finale)
def save_candidate(cand_dir, name: str, target: str, kind: str, description: str, val_metrics: dict,
                   predict_source: str, extra_files: dict | None = None):
    """Salva un candidato congelabile.

    cand_dir/candidate.json   metadati
    cand_dir/candidate.py     deve definire predict(df, lib) -> DataFrame con colonne 'score' e 'in_scope'
    cand_dir/<extra_files>    modelli (joblib), parametri, ecc.
    """
    cand_dir = Path(cand_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "candidate.py").write_text(predict_source)
    for fname, obj in (extra_files or {}).items():
        import joblib
        joblib.dump(obj, cand_dir / fname)
    (cand_dir / "candidate.json").write_text(json.dumps(
        {"name": name, "target": target, "kind": kind, "description": description, "val_metrics": val_metrics}, indent=2))
    return cand_dir
