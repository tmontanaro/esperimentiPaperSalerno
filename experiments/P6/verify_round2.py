"""P6 round 2: verifica indipendente delle revisioni R1-P3/R1-P4/R1-P5. Solo aggregati.
Uso: python3 experiments/P6/verify_round2.py [debug]"""
import sys, json, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from lib import echonext_lib as lib

debug = len(sys.argv) > 1 and sys.argv[1] == "debug"
OUT = ROOT / "results" / "P6" / ("debug" if debug else ""); OUT.mkdir(parents=True, exist_ok=True)
res = {}

# 1) registro: configurazioni per fase
reg = pd.read_csv(ROOT / "results/registry.csv")
res["registry_cols"] = list(reg.columns)
ncol = [c for c in reg.columns if "config" in c.lower()][0]
ucol = [c for c in reg.columns if "val" in c.lower()][0]
g = reg.groupby("phase").agg(n_exp=(ncol, "size"), n_configs=(ncol, "sum"))
g["configs_uses_val"] = reg[reg[ucol].astype(str).str.lower().isin(["true", "1", "yes"])].groupby("phase")[ncol].sum()
res["registry"] = g.fillna(0).astype(int).to_dict(orient="index")

# 2) R1-P5: conteggio confronti appaiati da local_vs_global.csv
lvg = pd.read_csv(ROOT / "results/P5/local_vs_global.csv")
res["lvg_cols"] = list(lvg.columns)
lo = [c for c in lvg.columns if c.lower() in ("diff_lo", "delta_lo", "d_lo")]
hi = [c for c in lvg.columns if c.lower() in ("diff_hi", "delta_hi", "d_hi")]
if lo and hi:
    d = lvg.dropna(subset=[lo[0], hi[0]])
    res["p5_paired"] = dict(n=int(len(d)), pos=int((d[lo[0]] > 0).sum()), neg=int((d[hi[0]] < 0).sum()))

# 3) R1-P4: ricalcolo indipendente stratificato (seed diverso)
data = lib.load_data(debug_n=2000 if debug else None)
val = lib.add_groups(data.val).reset_index(drop=True)
z = np.load(ROOT / "results/P4/val_pred_emb.npz", allow_pickle=True)
T = lib.TARGET
if not debug:
    assert (z["row_in_split"] == val.row_in_split.values).all()
    cnn = z["pred"][:, list(z["targets"]).index(T)].astype(float)
    if cnn.min() < 0 or cnn.max() > 1: cnn = 1 / (1 + np.exp(-cnn))
    spec = importlib.util.spec_from_file_location("p1", ROOT / "results/P1/candidates/global_best/candidate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tab = m.predict(val, lib)["score"].values
    y = val[T].values.astype(int)
    rows = []
    mg = lib.metrics_binary(y, cnn, 500); rows.append(dict(var="ALL", lev="ALL", n=len(y), pos=int(y.sum()), cnn=mg["auroc"], cnn_lo=mg["auroc_lo"], cnn_hi=mg["auroc_hi"], tab=lib.safe_auc(y, tab)))
    rng = np.random.default_rng(7)
    for col in ["race_ethnicity", "location_setting", "sex", "age_bin"]:
        for lev, idx in val.groupby(col, dropna=False).groups.items():
            ii = np.asarray(idx); yy = y[ii]
            if yy.sum() < 10 or (len(yy) - yy.sum()) < 10: continue
            mc = lib.metrics_binary(yy, cnn[ii], 500)
            dd = []
            for _ in range(500):
                b = rng.integers(0, len(ii), len(ii))
                if yy[b].min() == yy[b].max(): continue
                dd.append(lib.safe_auc(yy[b], cnn[ii][b]) - lib.safe_auc(yy[b], tab[ii][b]))
            rows.append(dict(var=col, lev=str(lev), n=len(ii), pos=int(yy.sum()), cnn=mc["auroc"], cnn_lo=mc["auroc_lo"], cnn_hi=mc["auroc_hi"],
                             tab=lib.safe_auc(yy, tab[ii]), d_lo=float(np.percentile(dd, 2.5)), d_hi=float(np.percentile(dd, 97.5))))
    S = pd.DataFrame(rows); S.to_csv(OUT / "strat_recalc.csv", index=False)
    # calibrazione grezza del composito (media predetta vs prevalenza)
    res["calib"] = dict(prev_val=float(y.mean()), mean_cnn=float(cnn.mean()), mean_tab=float(tab.mean()))

# 4) contratto dei candidati su un piccolo campione di val (seed diverso da R1-P4)
sub = val.sample(min(300, len(val)), random_state=123)
for name, cd in [("cnn_multilabel", "results/P4/candidates/cnn_multilabel"),
                 ("lvef_lte45_global", "results/P3/candidates/lvef_lte45_global"),
                 ("global_best", "results/P1/candidates/global_best")]:
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / cd / "candidate.py")
        mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
        r = mm.predict(sub.copy(), lib)
        info = dict(cols=list(r.columns), index_equal=bool(r.index.equals(sub.index)), n=len(r),
                    in_scope_frac=float(r["in_scope"].mean()), score_min=float(r["score"].min()), score_max=float(r["score"].max()),
                    nan_scores=int(r["score"].isna().sum()))
        if name == "cnn_multilabel" and not debug:
            pos = val.index.get_indexer(sub.index)
            info["max_abs_diff_vs_saved"] = float(np.abs(r["score"].values - cnn[pos]).max())
        res["cand_" + name] = info
    except Exception as e:
        res["cand_" + name] = dict(error=repr(e))
    if (ROOT / cd / "candidate.json").exists():
        res["cand_" + name]["json"] = json.load(open(ROOT / cd / "candidate.json"))

# 5) LVEF<=45 candidato su tutta la val
if not debug:
    spec = importlib.util.spec_from_file_location("lv", ROOT / "results/P3/candidates/lvef_lte45_global/candidate.py")
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    vv = val[val.lvef_lte_45_flag.notna()]
    r = mm.predict(vv, lib)
    res["lvef45_full_val"] = lib.metrics_binary(vv.lvef_lte_45_flag.values.astype(int), r["score"].values, 500)
    res["lvef45_full_val"]["n"] = int(len(vv)); res["lvef45_full_val"]["pos"] = int(vv.lvef_lte_45_flag.sum())


# 6) coerenza etichetta lvef_lte_45_flag vs lvef_value (aggregati)
for nm, D in (("train", data.train), ("val", data.val)):
    has = D.lvef_value.notna()
    res["lvef_flag_"+nm] = dict(n=int(len(D)), n_flag=int(D.lvef_lte_45_flag.notna().sum()), n_no_lvef=int((~has).sum()),
        flag_prev_no_lvef=float(D.loc[~has, "lvef_lte_45_flag"].mean()) if (~has).any() else None,
        agree_with_value=float(((D.loc[has, "lvef_value"] <= 45).astype(int) == D.loc[has, "lvef_lte_45_flag"]).mean()))

json.dump(res, open(OUT / "verify_round2.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:6000])
if not debug: lib.show(S.round(3))
