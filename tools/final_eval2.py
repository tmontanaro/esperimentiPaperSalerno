#!/usr/bin/env python3
"""SECONDO test, pre-specificato, UNA SOLA VOLTA (fase Q8). Legge agentic/FROZEN2.json:
  {"candidates": [{"name": "...", "dir": "results/Q6/candidates/cnn_shd", "target": "shd_moderate_or_greater_flag",
                   "comparator_dir": "results/P1/candidates/global_best" (opzionale)}], ...}
Prima di aprire il test verifica che le impronte (sha256) di candidati e pesi coincidano con quelle del congelamento (Q6).
Il test era già stato usato una volta nella pipeline 1 (P7): qui si dichiara come SECONDA lettura con pochi candidati scelti
SOLO con la cross-validation su train+validation. Non si può ripetere. Scrive results/final2/test_results.json e reports/final_test2_table.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "tools"))
import state2  # noqa: E402,F401  (stato della pipeline 2)
import state as S  # noqa: E402


def load_predict(cdir: Path):
    spec = importlib.util.spec_from_file_location(f"cand2_{cdir.name}", cdir / "candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict


def sha_dir(d: Path) -> dict:
    return {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.rglob("*")) if f.is_file() and "__pycache__" not in f.parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=str(ROOT / "agentic" / "FROZEN2.json"))
    ap.add_argument("--data-dir")
    ap.add_argument("--debug-n", type=int)
    a = ap.parse_args()

    out_json = ROOT / "results" / "final2" / "test_results.json"
    st = S.load()
    if (out_json.exists() or st.get("test_used")) and not a.debug_n:
        sys.exit("Il secondo test è già stato usato (results/final2/test_results.json esiste). Nessuna riesecuzione.")
    if not st.get("frozen"):
        sys.exit("Candidati non congelati: esegui `python3 tools/state2.py freeze` (fase Q6).")
    q8 = S.phase(st, "Q8")
    if not q8.get("approved") and not a.debug_n:
        sys.exit("Fase Q8 non approvata dall'umano: `python3 tools/state2.py approve Q8`.")
    fz = json.loads(Path(a.frozen).read_text())
    cands = fz.get("candidates", [])
    if not cands:
        sys.exit("FROZEN2.json non contiene candidati: nulla da valutare (salta Q8).")
    for c in cands:   # i candidati non devono essere cambiati dopo il congelamento
        if sha_dir(ROOT / c["dir"]) != c.get("sha256"):
            sys.exit(f"Il candidato {c['name']} è stato modificato dopo il congelamento: impronta diversa. Stop.")
        if c.get("weights_dir") and sha_dir(ROOT / c["weights_dir"]) != c.get("weights_sha256"):
            sys.exit(f"I pesi di {c['name']} sono stati modificati dopo il congelamento: impronta diversa. Stop.")

    os.environ["ECHONEXT_ALLOW_TEST"] = "1"
    import numpy as np
    import echonext_lib as lib
    import pipe2_lib as P2
    data = lib.load_data(a.data_dir, debug_n=a.debug_n, allow_test=True)
    test = lib.add_groups(data.test)
    results = {"n_test_rows": int(len(test)), "second_look": True, "candidates": []}
    for c in cands:
        cdir = ROOT / c["dir"]
        meta = json.loads((cdir / "candidate.json").read_text())
        tgt = c.get("target") or meta["target"]
        d = test.dropna(subset=[tgt])
        y = d[tgt].values.astype(int)
        res = load_predict(cdir)(d, lib)
        score, scope = res["score"].values, res["in_scope"].values.astype(bool)
        g = d.patient_key.values
        e = {"name": c["name"], "dir": c["dir"], "target": tgt, "val_metrics": meta.get("val_metrics")}
        if scope.sum() >= 20 and len(set(y[scope])) == 2:
            a_, lo, hi = P2.auc_ci_cluster(y[scope], score[scope], g[scope])
            e["test_in_scope"] = {"auroc": a_, "lo": lo, "hi": hi, "n": int(scope.sum()), "n_pos": int(y[scope].sum()),
                                  "auprc": float(lib.metrics_binary(y[scope], score[scope], n_boot=50)["auprc"]),
                                  "brier": P2.brier(y[scope], score[scope]), "ece": P2.ece(y[scope], score[scope])}
        known = d["lvef_value"].notna().values & scope
        if "lvef_value" in d.columns and known.sum() >= 20 and len(set(y[known])) == 2:
            a_, lo, hi = P2.auc_ci_cluster(y[known], score[known], g[known])
            e["test_lvef_known"] = {"auroc": a_, "lo": lo, "hi": hi, "n": int(known.sum()), "n_pos": int(y[known].sum())}
        if c.get("comparator_dir"):
            cs = load_predict(ROOT / c["comparator_dir"])(d, lib)["score"].values
            m = scope & ~np.isnan(cs)
            if m.sum() >= 20 and len(set(y[m])) == 2:
                a_, lo, hi = P2.auc_ci_cluster(y[m], cs[m], g[m])
                dd, dlo, dhi = P2.auc_diff_ci_cluster(y[m], score[m], cs[m], g[m])
                e["comparator"] = {"dir": c["comparator_dir"], "auroc": a_, "lo": lo, "hi": hi, "diff": dd, "diff_lo": dlo, "diff_hi": dhi}
        results["candidates"].append(e)

    if a.debug_n:
        print(json.dumps(results, indent=1, default=float)[:1500])
        print("(--debug-n: nulla è stato scritto né segnato come usato)")
        return
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=float))

    def f(m):
        return "n/d" if not m else f"{m['auroc']:.3f} [{m['lo']:.3f}-{m['hi']:.3f}] (n={m['n']}, pos={m['n_pos']})"
    lines = ["| Candidato | Target | Test (scope) | Test, righe con LVEF noto | Confronto (tabulare) | Differenza vs confronto | AUROC CV dev (da report) |", "|---|---|---|---|---|---|---|"]
    for e in results["candidates"]:
        cm = e.get("comparator")
        cmp_txt = "n/d" if not cm else f"{cm['auroc']:.3f} [{cm['lo']:.3f}-{cm['hi']:.3f}]"
        df_txt = "n/d" if not cm else f"{cm['diff']:+.3f} [{cm['diff_lo']:+.3f}, {cm['diff_hi']:+.3f}]"
        v = (e.get("val_metrics") or {}).get("auroc")
        lines.append(f"| {e['name']} | {e['target']} | {f(e.get('test_in_scope'))} | {f(e.get('test_lvef_known'))} | {cmp_txt} | {df_txt} | {v if v is None else round(v, 3)} |")
    (ROOT / "reports" / "final_test2_table.md").write_text("\n".join(lines) + "\n")
    with S.locked():
        st = S.load(); st["test_used"] = True; S.save(st)
    S.append_decision("decision", "Q8", f"secondo test usato una volta su {len(cands)} candidati congelati")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
