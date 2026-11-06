#!/usr/bin/env python3
"""Finto `claude` per collaudare il CABLAGGIO della pipeline senza spendere token.

Uso:  python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py
Simula gli agenti: scrive report validi (con numeri segnaposto, marcati SIMULATO) e, in P1, un candidato reale.
Iniezione di guasti (variabili d'ambiente):
  FAKE_FAIL_ONCE=P2     la fase P2 esce con errore la prima volta
  FAKE_LIMIT_ON=P3      la fase P3 simula un limite d'uso la prima volta
  FAKE_BAD_REPORT=P4    la fase P4 scrive un report incompleto la prima volta (test della riparazione)
I risultati NON hanno alcun valore scientifico.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
argv = sys.argv[1:]
prompt = argv[argv.index("-p") + 1] if "-p" in argv else ""
agent = argv[argv.index("--agent") + 1] if "--agent" in argv else ""


def out(text="simulato", error=False):
    print(json.dumps({"type": "result", "result": text, "session_id": "fake-session", "total_cost_usd": 0.0, "is_error": error}))
    sys.exit(1 if error else 0)


def once(env_name, pid):
    """True solo la prima volta che la fase `pid` incontra la condizione `env_name`."""
    if os.environ.get(env_name) != pid:
        return False
    marker = ROOT / "agentic" / f".fake_{env_name}_{pid}"
    if marker.exists():
        return False
    marker.write_text("1")
    return True


def run(*cmd):
    return subprocess.run([sys.executable, str(ROOT / "tools" / cmd[0]), *cmd[1:]], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "ECHONEXT_DIR": os.environ.get("ECHONEXT_DIR", "")})


def report(path, pid, name, extra_results="", status="OK", complete=True):
    body = f"""# Report {pid} · {name}
- Agente: {agent} (SIMULATO)

## Obiettivo
Obiettivo simulato della fase {pid}.

## Input letti
Handoff delle dipendenze (simulati).

## Cosa è stato fatto
Passi simulati; nessuna analisi reale.
"""
    if complete:
        body += f"""
## Risultati
AUROC simulato 0.600 [0.55-0.65], n=1000. {extra_results}

## Decisioni e motivazioni
Nessuna decisione reale (simulazione).

## Configurazioni provate
3 configurazioni simulate.

## Artefatti
results/{pid}/ (vuoto)

## Problemi aperti
Nessuno (simulazione).
"""
    body += f"""
## Handoff
STATUS: {status}
RISULTATI_CHIAVE:
- risultato simulato per {pid}
RACCOMANDAZIONE:
- procedere (simulato)
"""
    Path(ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / path).write_text(body)


PREDICT_SRC = '''import joblib
from pathlib import Path
import pandas as pd
_M = joblib.load(Path(__file__).parent / "model.joblib")


def predict(df, lib):
    X = df[lib.ECG_FEATURES].astype(float)
    return pd.DataFrame({"score": _M.predict_proba(X)[:, 1], "in_scope": True}, index=df.index)
'''

# ------------------------------------------------------------------ orchestratore (gate)
if agent == "orchestrator":
    m = re.search(r"GATE dopo la fase (\S+)", prompt)
    pid = m.group(1) if m else "?"
    if pid == "P6":
        cd = "results/P1/candidates/global_hist_gb"
        if (ROOT / cd / "candidate.py").exists():
            (ROOT / "agentic" / "FROZEN.json").write_text(json.dumps(
                {"comparator_dir": cd, "candidates": [{"name": "global_hist_gb", "dir": cd, "target": "shd_moderate_or_greater_flag"}]}))
            run("state.py", "freeze")
    run("state.py", "note", "--kind", "decision", "--phase", pid, "--text", f"gate {pid} simulato: nessuna modifica")
    out(f"gate {pid} simulato")

# ------------------------------------------------------------------ riparazione report
if prompt.startswith("Il report"):
    path = re.search(r"Il report (\S+) non", prompt).group(1)
    pid = Path(path).name.split("_")[0]
    report(path, pid, Path(path).stem.split("_", 1)[1])
    out("report riparato")

# ------------------------------------------------------------------ agenti di fase
m = re.search(r"fase (\S+) \((\S+)\)", prompt)
pid, name = m.group(1), m.group(2)
path = re.search(r"Scrivi il report in (\S+) ", prompt).group(1)

if once("FAKE_FAIL_ONCE", pid):
    out("errore simulato", error=True)
if once("FAKE_LIMIT_ON", pid):
    print("Claude usage limit reached. Your limit will reset at 21:00", file=sys.stderr)
    sys.exit(1)

if pid == "P1":  # candidato reale, usato poi dalla valutazione finale
    sys.path.insert(0, str(ROOT / "lib"))
    import echonext_lib as lib
    d = lib.load_data(debug_n=int(os.environ.get("FAKE_DEBUG_N", "0")) or None)
    Xtr, ytr, _ = lib.get_xy(d.train, lib.TARGET)
    Xva, yva, _ = lib.get_xy(d.val, lib.TARGET)
    m_ = lib.model_zoo()["hist_gb"].fit(Xtr, ytr)
    vm = lib.metrics_binary(yva, m_.predict_proba(Xva)[:, 1])
    lib.save_candidate(ROOT / "results/P1/candidates/global_hist_gb", "global_hist_gb", lib.TARGET, "global",
                       "HistGradientBoosting su tutte le feature ECG", vm, PREDICT_SRC, {"model.joblib": m_})
    run("state.py", "log-exp", "--phase", "P1", "--desc", "hist_gb globale", "--uses-val", "--n-configs", "5")
    report(path, pid, name, f"AUROC val hist_gb = {vm['auroc']:.3f}.")
elif pid == "P7":
    r = run("final_eval.py")
    report(path, pid, name, "Tabella test in reports/final_test_table.md. " + r.stdout[-200:].replace("\n", " "))
elif once("FAKE_BAD_REPORT", pid):
    report(path, pid, name, complete=False)
else:
    report(path, pid, name)
out(f"fase {pid} simulata")
