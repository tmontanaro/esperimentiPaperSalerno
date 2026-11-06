#!/usr/bin/env python3
"""Valutazione finale sul test set: si esegue UNA SOLA VOLTA, solo su candidati congelati.

Legge agentic/FROZEN.json:
  {"comparator_dir": "results/P1/candidates/global_hist_gb",
   "candidates": [{"name": "...", "dir": "results/P2/candidates/sub_qrs", "target": "lvef_lte_45_flag"}, ...]}

Ogni cartella candidata contiene candidate.py con  predict(df, lib) -> DataFrame[score, in_scope]  (indice come df).
Scrive results/final/test_results.json e reports/final_test_table.md. Rifiuta di rieseguire.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "tools"))


def load_predict(cdir: Path):
    spec = importlib.util.spec_from_file_location(f"cand_{cdir.name}", cdir / "candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=str(ROOT / "agentic" / "FROZEN.json"))
    ap.add_argument("--data-dir")
    ap.add_argument("--debug-n", type=int)
    ap.add_argument("--allow-rerun", action="store_true", help="solo l'umano, e annotando il motivo in DECISIONS.md")
    a = ap.parse_args()

    out_json = ROOT / "results" / "final" / "test_results.json"
    if out_json.exists() and not a.allow_rerun:
        sys.exit("Il test set è già stato usato (results/final/test_results.json esiste). Nessuna riesecuzione.")
    fz = json.loads(Path(a.frozen).read_text())
    if not fz.get("candidates"):
        sys.exit("FROZEN.json non contiene candidati: nulla da valutare (salta P7).")

    os.environ["ECHONEXT_ALLOW_TEST"] = "1"
    import echonext_lib as lib
    data = lib.load_data(a.data_dir, debug_n=a.debug_n, allow_test=True)
    test = lib.add_groups(data.test)

    cands = list(fz["candidates"])
    comp = fz.get("comparator_dir")
    results = {"n_test_rows": int(len(test)), "candidates": []}
    comp_fn = load_predict(ROOT / comp) if comp else None

    for c in cands:
        cdir = ROOT / c["dir"]
        meta = json.loads((cdir / "candidate.json").read_text())
        tgt = c.get("target") or meta["target"]
        d = test.dropna(subset=[tgt])
        y = d[tgt].values
        res = load_predict(cdir)(d, lib)
        score, scope = res["score"].values, res["in_scope"].values.astype(bool)
        entry = {"name": c["name"], "dir": c["dir"], "target": tgt, "kind": meta.get("kind"),
                 "val_metrics": meta.get("val_metrics"), "test_in_scope": None, "test_all": None, "comparator_in_scope": None, "test_lvef_known": None}
        if scope.sum() >= 20 and len(set(y[scope])) == 2:
            entry["test_in_scope"] = lib.metrics_binary(y[scope], score[scope])
        entry["test_all"] = lib.metrics_binary(y, score) if len(set(y)) == 2 else None
        # metrica pre-specificata dal revisore: solo righe con lvef_value noto (le altre hanno tutte le etichette = 0 per costruzione)
        if "lvef_value" in d.columns:
            known = d["lvef_value"].notna().values
            if known.sum() >= 20 and len(set(y[known])) == 2:
                entry["test_lvef_known"] = lib.metrics_binary(y[known], score[known])
        if comp_fn is not None:
            cr = comp_fn(d, lib)
            cs = cr["score"].values
            if scope.sum() >= 20 and len(set(y[scope])) == 2:
                entry["comparator_in_scope"] = lib.metrics_binary(y[scope], cs[scope])
        results["candidates"].append(entry)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    def f(m):
        return "n/d" if not m else f"{m['auroc']:.3f} [{m['auroc_lo']:.3f}-{m['auroc_hi']:.3f}] (n={m['n']}, pos={m['n_pos']})"
    lines = ["| Candidato | Target | Test nello scope | Globale sugli stessi pazienti | Test su tutti | Test su righe con LVEF noto | AUROC val (da report) |", "|---|---|---|---|---|---|---|"]
    for e in results["candidates"]:
        v = (e.get("val_metrics") or {}).get("auroc")
        lines.append(f"| {e['name']} | {e['target']} | {f(e['test_in_scope'])} | {f(e['comparator_in_scope'])} | {f(e['test_all'])} | {f(e['test_lvef_known'])} | {v if v is None else round(v, 3)} |")
    (ROOT / "reports" / "final_test_table.md").write_text("\n".join(lines) + "\n")

    import state  # segna il test come usato
    with state.locked():
        st = state.load()
        st["test_used"] = True
        state.save(st)
    state.append_decision("decision", "P7", f"test set usato una volta su {len(cands)} candidati")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
