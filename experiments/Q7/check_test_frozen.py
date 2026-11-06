"""Q7: controlli 4 e 7 (test chiuso, impronte sha256, ordine temporale preregistrazione -> pesi -> freeze, contratto dei candidati).
Non carica dati, non modifica nulla. Stampa solo aggregati/metadati di file."""
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
res = {}
ts = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))

# 4. test chiuso
res["final2_exists"] = (ROOT / "results/final2").exists()
hits = []
for p in sorted((ROOT / "experiments").glob("Q*/**/*.py")):
    if p.parent.name == "Q7":
        continue
    txt = p.read_text(errors="ignore")
    for pat in ("allow_test", "test_results", "final_eval", "split == \"test\"", "split=='test'", "'test'", '"test"'):
        if pat in txt:
            hits.append([str(p.relative_to(ROOT)), pat])
res["experiments_Q_test_hits"] = hits
res["final_dir_files"] = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "results/final").rglob("*"))
res["final_test_results_mtime"] = ts(ROOT / "results/final/test_results.json")
res["final_eval2_exists"] = (ROOT / "tools/final_eval2.py").exists()

# 7. impronte
F = json.loads((ROOT / "agentic/FROZEN2.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bad, n = [], 0
for c in F["candidates"]:
    for fn, h in c["sha256"].items():
        n += 1
        if sha(ROOT / c["dir"] / fn) != h:
            bad.append([c["name"], fn])
    for fn, h in c["weights_sha256"].items():
        n += 1
        if sha(ROOT / c["weights_dir"] / fn) != h:
            bad.append([c["name"], c["weights_dir"], fn])
    # weights dir referenced in candidate.py must equal weights_dir, target must match
    py = (ROOT / c["dir"] / "candidate.py").read_text()
    w = re.search(r'WEIGHTS = ROOT / "([^"]+)"', py).group(1)
    t = re.search(r'TARGET = "([^"]+)"', py).group(1)
    cj = json.loads((ROOT / c["dir"] / "candidate.json").read_text())
    c["_check"] = {"py_weights_ok": w == c["weights_dir"], "py_target_ok": t == c["target"], "json_target_ok": cj["target"] == c["target"],
                   "cv_auroc": cj["val_metrics"]["auroc"], "comparator_exists": (ROOT / c["comparator_dir"]).exists() if "comparator_dir" in c else None,
                   "has_predict": "def predict(df, lib)" in py}
res["n_hashes_checked"] = n
res["hash_mismatch"] = bad
res["candidate_checks"] = {c["name"]: c["_check"] for c in F["candidates"]}
res["n_candidates"] = len(F["candidates"])
# leads in final-fit configs
for d in ("final_all", "final_I_II_V1"):
    cfg = json.loads((ROOT / "results/Q6" / d / "config.json").read_text())
    res[f"{d}_cfg"] = {k: cfg.get(k) for k in ("leads", "epochs", "lr", "reg_weight", "batch", "population", "seeds")}
q1 = json.loads((ROOT / "results/Q1/main/config.json").read_text())
q2 = json.loads((ROOT / "results/Q2/I_II_V1/config.json").read_text())
res["q2_I_II_V1_leads"] = q2["leads"]
res["q1_cfg"] = {k: q1.get(k) for k in ("leads", "epochs", "lr", "reg_weight", "batch", "population")}
# tempi
res["mtime"] = {"preregistration.md": ts(ROOT / "results/Q6/preregistration.md")}
for d in ("final_all", "final_I_II_V1"):
    for p in sorted((ROOT / "results/Q6" / d).glob("*")):
        res["mtime"][f"{d}/{p.name}"] = ts(p)
for c in F["candidates"]:
    for fn in ("candidate.json", "candidate.py"):
        res["mtime"][f"{c['name']}/{fn}"] = ts(ROOT / c["dir"] / fn)
res["frozen_at"] = F.get("frozen_at")
res["FROZEN2_mtime"] = ts(ROOT / "agentic/FROZEN2.json")
print(json.dumps(res, indent=1))
(ROOT / "results/Q7").mkdir(exist_ok=True)
(ROOT / "results/Q7/check_test_frozen.json").write_text(json.dumps(res, indent=1))
