#!/usr/bin/env python3
"""Driver della pipeline multi-agente (modalità non interattiva, `claude -p`).

Ogni fase gira in una sessione NUOVA di Claude Code con l'agente indicato; tutto lo stato vive su disco
(agentic/pipeline_state.json, reports/, results/), quindi si può interrompere e riprendere in qualsiasi momento.

Codici di uscita:  0 = pipeline finita  |  2 = fase fallita o bloccata  |  3 = cartella non approvata in Claude Code  |  10 = in attesa di approvazione umana
                   75 = limite d'uso/rate limit raggiunto (rilancia più tardi: riparte da dove era)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import state as S  # noqa: E402
import validate_report as V  # noqa: E402

RUNLOG = ROOT / "agentic" / "RUNLOG.md"
LIMIT_RE = re.compile(r"(usage limit|rate.?limit|limit reached|quota|credit balance|overloaded|too many requests|\b429\b|\b529\b|resets? at)", re.I)

PHASE_PROMPT = """Sei l'agente «{agent}» e stai eseguendo la fase {id} ({name}) della pipeline EchoNext.
Obiettivo breve: {goal}
Istruzioni dettagliate: la sezione «{id}» di agentic/PLAYBOOK.md e il tuo file di agente (già caricato).
Fasi da cui dipendi: {deps}.

Inizia con:  python3 tools/state.py brief   e, per ogni dipendenza,  python3 tools/state.py handoff <ID>
{resume_note}
Scrivi il report in {report} seguendo agentic/REPORT_TEMPLATE.md e convalidalo con:
  python3 tools/validate_report.py {report}
NON segnare la fase come completata (lo fa il driver). Non aprire il test set. Non stampare dati a livello di paziente."""

GATE_PROMPT = """GATE dopo la fase {id} ({name}).
1) python3 tools/state.py brief
2) python3 tools/state.py handoff {id}
3) Applica la sezione «Gate {id}» di agentic/PLAYBOOK.md ed esegui le azioni SOLO con tools/state.py
   (skip, add-phase, set-config, set-priority, note, freeze...). Non eseguire analisi.
4) Registra ogni decisione con: python3 tools/state.py note --kind decision --phase {id} --text "..."
Rispondi con al massimo 6 righe: decisione, azioni eseguite, motivo."""


BACKUPS = ROOT / "agentic" / "backups"


def snapshot(label: str) -> None:
    """Copia di sicurezza dello stato prima/dopo ogni fase (piccola: stato, decisioni, log, registro esperimenti)."""
    import shutil
    d = BACKUPS / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}"
    d.mkdir(parents=True, exist_ok=True)
    for f in (S.STATE, S.DECISIONS, RUNLOG, ROOT / "results" / "registry.csv"):
        if f.exists():
            shutil.copy2(f, d / f.name)


def git_checkpoint(label: str) -> None:
    """Se il progetto è un repository git, salva un commit dopo ogni fase (best effort, non blocca mai)."""
    if not (ROOT / ".git").exists():
        return
    try:
        subprocess.run(["git", "-C", str(ROOT), "add", "-A"], capture_output=True, timeout=120)
        subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", f"checkpoint: {label}"], capture_output=True, timeout=120)
    except Exception:
        pass


class Untrusted(Exception):
    """Claude Code ignora i permessi del progetto finché la cartella non è stata approvata in modo interattivo."""


UNTRUSTED_MSG = ("Claude Code non si fida ancora di questa cartella: in modalità non interattiva ignora i permessi del progetto "
                 "(.claude/settings.json) e gli agenti non possono scrivere né lanciare Python.\n"
                 "Rimedio (una volta sola): apri `claude` in questa cartella, accetta la richiesta di fiducia, esci con /exit e rilancia.")


def log(line: str) -> None:
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    if not RUNLOG.exists():
        RUNLOG.write_text("# Log delle esecuzioni (append-only)\n\n| Quando | Fase | Agente | Esito | Durata | Costo stimato | Note |\n|---|---|---|---|---|---|---|\n")
    with open(RUNLOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def call_claude(agent: str, prompt: str, a, cfg: dict) -> dict:
    cmd = [a.claude_bin, "-p", prompt, "--agent", agent, "--output-format", "json",
           "--permission-mode", a.permission_mode, "--max-turns", str(a.max_turns),
           "--max-budget-usd", str(a.max_budget_usd)]
    if a.model:
        cmd += ["--model", a.model]
    env = os.environ.copy()
    env["ECHONEXT_DIR"] = cfg.get("data_dir") or ""
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=a.timeout_min * 60, env=env)
    except subprocess.TimeoutExpired:
        return dict(ok=False, limit=False, secs=time.time() - t0, cost=None, note=f"timeout dopo {a.timeout_min} min", result="")
    out, err = p.stdout or "", p.stderr or ""
    if "has not been trusted" in err or "has not been trusted" in out[:400]:
        raise Untrusted()
    data = {}
    try:
        data = json.loads(out)
        if isinstance(data, list):
            data = next((x for x in reversed(data) if isinstance(x, dict) and x.get("type") == "result"), {})
    except Exception:
        pass
    is_error = bool(data.get("is_error")) if isinstance(data, dict) else False
    ok = p.returncode == 0 and not is_error
    text = (data.get("result") if isinstance(data, dict) else None) or out[-500:]
    limit = (not ok) and bool(LIMIT_RE.search(out + "\n" + err))
    note = "" if ok else f"rc={p.returncode} {data.get('subtype', '') if isinstance(data, dict) else ''} {(err or text)[-200:]}".strip()
    return dict(ok=ok, limit=limit, secs=time.time() - t0, cost=data.get("total_cost_usd") if isinstance(data, dict) else None,
                note=note.replace("\n", " ").replace("|", "/"), result=str(text))


def fmt(r: dict) -> tuple[str, str]:
    return f"{r['secs'] / 60:.1f} min", ("n/d" if r["cost"] is None else f"${float(r['cost']):.2f}")


def set_status(pid: str, status: str, summary: str = "", dec_attempt: bool = False) -> None:
    with S.locked():
        st = S.load()
        p = S.phase(st, pid)
        p["status"] = status
        p["finished"] = S.now() if status in ("done", "failed", "skipped") else None
        if summary:
            p["summary"] = summary
        if dec_attempt:
            p["attempts"] = max(0, p["attempts"] - 1)
        S.save(st)


def run_phase(st: dict, p: dict, a) -> str:
    cfg = st["config"]
    pid, rp = p["id"], f"reports/{p['id']}_{p['name']}.md"
    S.cmd_start(argparse.Namespace(id=pid))
    st = S.load()
    p = S.phase(st, pid)
    resume = ""
    if p["attempts"] > 1 or p.get("interrupted"):
        resume = (f"ATTENZIONE: questo è un RIAVVIO (tentativo {p['attempts']}). Leggi reports/{pid}_{p['name']}.progress.md e i file in "
                  f"results/{pid}/: riparti da lì senza rifare ciò che esiste già. Motivo dell'ultima interruzione: {p.get('summary') or 'sconosciuto'}.")
    extra = f"\nQuesta è una fase di FOLLOW-UP creata dall'orchestratore" + (f" per rivedere {p['revises']}" if p.get("revises") else "") + "." if p.get("kind") == "followup" else ""
    prompt = PHASE_PROMPT.format(agent=p["agent"], id=pid, name=p["name"], goal=p["goal"] + extra, report=rp,
                                 deps=", ".join(p["depends_on"]) or "nessuna", resume_note=resume)
    try:
        r = call_claude(p["agent"], prompt, a, cfg)
    except Untrusted:
        set_status(pid, "pending", "cartella non approvata in Claude Code", dec_attempt=True)
        raise
    dur, cost = fmt(r)
    if r["limit"]:
        set_status(pid, "pending", "interrotta da limite d'uso/rate limit", dec_attempt=True)
        log(f"| {S.now()} | {pid} | {p['agent']} | LIMITE | {dur} | {cost} | {r['note']} |")
        return "LIMIT"
    if not r["ok"]:
        if p["attempts"] < cfg["max_attempts"]:
            set_status(pid, "pending", r["note"])
            log(f"| {S.now()} | {pid} | {p['agent']} | RIPROVO | {dur} | {cost} | {r['note']} |")
            return "RETRY"
        S.cmd_fail(argparse.Namespace(id=pid, reason=r["note"] or "errore"))
        log(f"| {S.now()} | {pid} | {p['agent']} | FALLITA | {dur} | {cost} | {r['note']} |")
        return "FAILED"
    errs = V.check(ROOT / rp)
    if errs:  # un solo tentativo di riparazione del report
        fix = f"Il report {rp} non supera la validazione:\n- " + "\n- ".join(errs) + f"\nCorreggi SOLO il report (struttura in agentic/REPORT_TEMPLATE.md) e riesegui: python3 tools/validate_report.py {rp}"
        r2 = call_claude(p["agent"], fix, a, cfg)
        dur2, cost2 = fmt(r2)
        log(f"| {S.now()} | {pid} | {p['agent']} | RIPARAZIONE REPORT | {dur2} | {cost2} | {len(errs)} problemi |")
        errs = V.check(ROOT / rp)
    if errs:
        set_status(pid, "pending", "report non valido: " + "; ".join(errs)[:300])
        if p["attempts"] >= cfg["max_attempts"]:
            S.cmd_fail(argparse.Namespace(id=pid, reason="report non valido: " + "; ".join(errs)[:300]))
            log(f"| {S.now()} | {pid} | {p['agent']} | FALLITA | {dur} | {cost} | report non valido |")
            return "FAILED"
        log(f"| {S.now()} | {pid} | {p['agent']} | RIPROVO | {dur} | {cost} | report non valido |")
        return "RETRY"
    status = V.status_of(ROOT / rp)
    if status == "FALLITO":
        S.cmd_fail(argparse.Namespace(id=pid, reason="l'agente ha dichiarato STATUS: FALLITO (vedi report)"))
        log(f"| {S.now()} | {pid} | {p['agent']} | FALLITA (dichiarata) | {dur} | {cost} | vedi {rp} |")
        return "FAILED"
    S.cmd_done(argparse.Namespace(id=pid, summary=f"STATUS {status}"))
    log(f"| {S.now()} | {pid} | {p['agent']} | OK ({status}) | {dur} | {cost} | {rp} |")
    return "OK"


def run_gate(p: dict, a) -> str:
    st = S.load()
    r = call_claude("orchestrator", GATE_PROMPT.format(id=p["id"], name=p["name"]), a, st["config"])
    dur, cost = fmt(r)
    if r["limit"]:
        log(f"| {S.now()} | gate {p['id']} | orchestrator | LIMITE | {dur} | {cost} | {r['note']} |")
        return "LIMIT"
    log(f"| {S.now()} | gate {p['id']} | orchestrator | {'OK' if r['ok'] else 'ERRORE'} | {dur} | {cost} | {(r['result'] or r['note'])[:160].replace(chr(10), ' ')} |")
    return "OK"


def resolve_claude(name: str) -> str | None:
    """Trova l'eseguibile di Claude Code: PATH, poi posizioni di installazione comuni."""
    found = shutil.which(name)
    if found:
        return found
    if os.sep in name and os.access(name, os.X_OK):
        return name
    home = Path.home()
    for c in (home / ".local/bin/claude", home / ".claude/local/claude", home / ".npm-global/bin/claude",
              Path("/usr/local/bin/claude"), Path("/opt/homebrew/bin/claude")):
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


NOT_FOUND_MSG = ("Non trovo il programma `claude` (Claude Code). Il driver gira in Python e usa il PATH del suo ambiente.\n"
                 "  1. Verifica in questo stesso terminale: `which claude` e `claude --version`.\n"
                 "  2. Se non è installato: vedi il passo 1 di GUIDA.md.\n"
                 "  3. Se è installato ma non nel PATH: rilancia con `--claude-bin /percorso/di/claude` (lo mostra `which claude`\n"
                 "     da un terminale fuori dal venv) oppure aggiungi la cartella al PATH (di solito ~/.local/bin).\n"
                 "Nessuna fase è stata avviata; lo stato su disco è intatto.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--permission-mode", default="dontAsk")
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--max-budget-usd", type=float, default=10.0, help="tetto per ogni singola chiamata (stima lato client)")
    ap.add_argument("--timeout-min", type=int, default=240)
    ap.add_argument("--model", help="forza un modello per tutte le chiamate (es. sonnet)")
    ap.add_argument("--until", help="fermati dopo questa fase (es. P3)")
    ap.add_argument("--no-gates", action="store_true", help="salta le chiamate dell'orchestratore tra le fasi")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run:
        exe = resolve_claude(a.claude_bin)
        if not exe:
            print(NOT_FOUND_MSG, file=sys.stderr)
            return 4
        a.claude_bin = exe

    st = S.load()
    if not a.dry_run:
        S.cmd_recover(argparse.Namespace())
    while True:
        st = S.load()
        owed = [q for q in st["phases"] if q["status"] == "done" and q.get("gate_pending")]
        if owed and not a.no_gates and not a.dry_run:  # gate dovuto (anche se interrotto in un'esecuzione precedente)
            q = owed[0]
            if run_gate(q, a) == "LIMIT":
                print("\nLimite d'uso raggiunto durante il gate: rilancia più tardi, il gate verrà ripetuto.")
                return 75
            S.cmd_gate_done(argparse.Namespace(id=q["id"]))
            snapshot(f"dopo_gate_{q['id']}")
            git_checkpoint(f"gate {q['id']}")
            if a.until and q["id"] == a.until:
                print(f"Fermato dopo {a.until} come richiesto.")
                return 0
            continue
        todo = S.runnable(st)
        if not todo:
            failed = [p for p in st["phases"] if p["status"] == "failed"]
            pending = [p for p in st["phases"] if p["status"] == "pending"]
            if failed:
                print("Pipeline ferma: fasi fallite:", ", ".join(f"{p['id']} ({p['summary']})" for p in failed))
                print("Dopo aver risolto: python3 tools/state.py reset <ID> --attempts   e rilancia.")
                return 2
            if pending:
                print("Pipeline bloccata: dipendenze non soddisfatte per", ", ".join(p["id"] for p in pending))
                return 2
            print("Pipeline completata. Vedi reports/ (report finale: reports/P8_synthesis.md).")
            return 0
        p = todo[0]
        if p.get("requires_approval") and not p.get("approved"):
            print(f"\nPAUSA: la fase {p['id']} ({p['name']}) richiede la tua approvazione.\n"
                  f"Controlla agentic/FROZEN.json e reports/P6_review.md; poi:  python3 tools/state.py approve {p['id']}  e rilancia.")
            return 10
        if a.dry_run:
            print("Prossima fase:", p["id"], p["name"], p["agent"])
            print("Comando:", a.claude_bin, "-p <prompt>", "--agent", p["agent"], "--permission-mode", a.permission_mode,
                  "--max-turns", a.max_turns, "--max-budget-usd", a.max_budget_usd)
            return 0
        snapshot(f"prima_{p['id']}")
        outcome = run_phase(st, p, a)
        snapshot(f"dopo_{p['id']}_{outcome}")
        git_checkpoint(f"{p['id']} {outcome}")
        if outcome == "LIMIT":
            print("\nLimite d'uso raggiunto. Lo stato è salvato: rilancia più tardi con lo stesso comando.")
            return 75
        if outcome == "FAILED":
            continue  # il ciclo successivo si ferma con il messaggio sulle fasi fallite
        if a.until and p["id"] == a.until and outcome == "OK" and (a.no_gates or not p.get("gate")):
            print(f"Fermato dopo {a.until} come richiesto.")
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Untrusted:
        print("\n" + UNTRUSTED_MSG)
        sys.exit(3)
