"""Verifica del notebook P9. nbclient/ipykernel NON sono installati -> si esegue ogni cella di codice, in ordine, in un unico
namespace Python (stesso effetto di un kernel, senza output salvati). Uso:
  python3 experiments/P9/run_check.py quick|full [heavy]
Salva/aggiorna results/P9/run_check.json. Non stampa dati di paziente (solo aggregati prodotti dal notebook; qui conta solo l'esito)."""
import ast, contextlib, io, json, os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "reports/P9_notebook/replica_analisi.ipynb"
OUTJ = ROOT / "results/P9/run_check.json"
mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
heavy = "heavy" in sys.argv[2:]
cfg = json.loads((ROOT / "agentic/pipeline_state.json").read_text())["config"]
os.environ["ECHONEXT_DIR"] = cfg["data_dir"]

nb = json.loads(NB.read_text())
src = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "il notebook ha output salvati"
for i, s in enumerate(src):
    ast.parse(s)  # controllo sintattico
print("ast.parse OK su", len(src), "celle di codice")

# parametri della prova (sostituiti solo nel testo eseguito, il file consegnato non cambia)
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
    if not ok: break
tot = round(time.time() - t00, 1)
key = f"{mode}{'_heavy' if heavy else ''}"
j = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
j[key] = {"celle_codice": len(src), "ok": sum(r["ok"] for r in res), "fallite": sum(not r["ok"] for r in res), "sec": tot, "dettaglio": res,
          "metodo": "esecuzione in-process delle celle (nbclient non installato)",
          "risultati": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ns.get("RESULTS", {}).items()}}
if not heavy:
    j[f"{mode}_total_sec"] = tot
OUTJ.parent.mkdir(parents=True, exist_ok=True); OUTJ.write_text(json.dumps(j, indent=1))
print("totale", tot, "s; celle ok:", j[key]["ok"], "/", len(src)); print(json.dumps(j[key]["risultati"]))
