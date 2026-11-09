"""Verifica del notebook Q10 (nbclient/ipykernel NON installati -> si esegue ogni cella di codice, in ordine, in un unico namespace).
Uso: python3 experiments/Q10/run_check2.py quick|full [heavy]
Salva/aggiorna results/Q10/run_check.json. Il test non viene caricato (il notebook scarta le righe di test in load_data)."""
import ast, contextlib, io, json, os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "reports/Q10_notebook/replica_completa.ipynb"
OUTJ = ROOT / "results/Q10/run_check.json"
mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
heavy = "heavy" in sys.argv[2:]
cfg = json.loads((ROOT / "agentic/pipeline_state.json").read_text())["config"]
os.environ["ECHONEXT_DIR"] = cfg["data_dir"]

nb = json.loads(NB.read_text())
src = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "il notebook ha output salvati"
for s in src:
    ast.parse(s)
print("ast.parse OK su", len(src), "celle di codice")
assert not any("allow_test" in s or "import lib" in s or "from lib" in s or "tools/" in s.split("#")[0] for s in src if "load_fold_model" in s), "dipendenze"
src[0] = src[0].replace("QUICK = True", f"QUICK = {mode == 'quick'}").replace("RUN_HEAVY = False", f"RUN_HEAVY = {heavy}")
ns, res, t00 = {"__name__": "__main__"}, [], time.time()
for i, s in enumerate(src):
    t0 = time.time(); buf = io.StringIO(); ok, err = True, None
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(s, f"cella_{i}", "exec"), ns)
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"; traceback.print_exc()
    out = buf.getvalue()
    res.append({"cella": i, "ok": ok, "sec": round(time.time() - t0, 1), "righe_output": out.count("\n"), "errore": err})
    print(f"cella {i}: {'ok' if ok else 'FALLITA'} {res[-1]['sec']}s")
    if not ok:
        break
tot = round(time.time() - t00, 1)
key = f"{mode}{'_heavy' if heavy else ''}"
j = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
chk = ns.get("CHECKS", [])
j[key] = {"celle_codice": len(src), "ok": sum(r["ok"] for r in res), "fallite": sum(not r["ok"] for r in res), "sec": tot, "dettaglio": res,
          "metodo": "esecuzione in-process delle celle (nbclient non installato)",
          "confronto_numeri": [[a, b, (round(float(c), 4) if c is not None else None)] for a, b, c in chk]}
OUTJ.parent.mkdir(parents=True, exist_ok=True); OUTJ.write_text(json.dumps(j, indent=1))
print("totale", tot, "s; celle ok:", j[key]["ok"], "/", len(src))
for a, b, c in chk:
    print(f"  {a}: atteso {b} ottenuto {float(c):.4f}")
