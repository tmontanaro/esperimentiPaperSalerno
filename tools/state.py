#!/usr/bin/env python3
"""Stato della pipeline su disco: l'unica fonte di verità per ripartire dopo un'interruzione.

Esempi:
  python3 tools/state.py init --data-dir /percorso/dati
  python3 tools/state.py brief                 # riepilogo da leggere all'inizio di ogni sessione
  python3 tools/state.py next                  # fasi eseguibili ora
  python3 tools/state.py handoff P1            # sezione Handoff del report di P1
  python3 tools/state.py note --kind decision --phase P2 --text "..."
  python3 tools/state.py log-exp --phase P2 --desc "..." --uses-val --n-configs 12
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AG = ROOT / "agentic"
STATE = AG / "pipeline_state.json"
LOCK = AG / ".state.lock"
DECISIONS = AG / "DECISIONS.md"
REG = ROOT / "results" / "registry.csv"
REPORTS = ROOT / "reports"

DEFAULT_CONFIG = {
    "data_dir": None,
    "primary_target": "shd_moderate_or_greater_flag",
    "max_followups": 4,
    "max_review_rounds": 2,
    "max_attempts": 3,
    "thresholds": {
        "tabular_signal_min_gain": 0.02,       # AUROC(tutte le feature) - AUROC(solo età+sesso)
        "promising_target_auroc_lo": 0.65,
        "credible_subgroup_margin": 0.02,
        "good_auroc": 0.80,
        "min_n_val": 100,
        "min_pos_val": 25,
    },
    "tabular_signal": None,
    "cluster_targets": [],
    "clusters_mode": "normal",
}

DEFAULT_PHASES = [
    dict(id="P0", name="data-audit", agent="data-auditor", depends_on=[], gate=True, priority=0,
         goal="Trovare i file, verificare l'integrità del dataset (split, pazienti, mancanti, forme d'onda) e profilare le etichette."),
    dict(id="P1", name="baseline", agent="baseline-modeler", depends_on=["P0"], gate=True, priority=0,
         goal="Baseline tabulare sul target composito: età+sesso contro tutte le feature ECG, 5 modelli, CV per paziente."),
    dict(id="P2", name="conditions", agent="condition-analyst", depends_on=["P1"], gate=True, priority=0,
         goal="Predizione per singola condizione, stratificazione per gruppi noti, scoperta di sottogruppi validata."),
    dict(id="P3", name="lvef-qrs", agent="lvef-qrs-analyst", depends_on=["P0", "P1"], gate=True, priority=0,
         goal="LVEF (regressione e <=45%) e ipotesi dei medici sul QRS come modificatore d'effetto."),
    dict(id="P4", name="waveform", agent="waveform-modeler", depends_on=["P0", "P1"], gate=True, priority=0,
         goal="CNN 1D multi-etichetta sulle forme d'onda, confronto con la baseline tabulare."),
    dict(id="P5", name="clusters", agent="cluster-analyst", depends_on=["P1", "P2"], gate=True, priority=0,
         goal="Cluster con selezione incrementale delle feature e predizione dentro i cluster, con soglia di rumore."),
    dict(id="P6", name="review", agent="skeptic-reviewer", depends_on=["P2", "P3", "P4", "P5"], gate=True, priority=0,
         goal="Revisione indipendente: leakage, riproduzione dei numeri, selezione sul validation, affermazioni non supportate."),
    dict(id="P7", name="final-test", agent="final-evaluator", depends_on=["P6"], gate=False, priority=0,
         requires_approval=True,
         goal="Valutazione UNA SOLA VOLTA dei candidati congelati sul test set e interpretazione."),
    dict(id="P8", name="synthesis", agent="synthesizer", depends_on=["P7"], gate=False, priority=0,
         goal="Report finale spiegato per i partner: risultati, limiti, cosa funziona e cosa no."),
]


# ------------------------------------------------------------------ utilità
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@contextlib.contextmanager
def locked():
    AG.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "w")
    try:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        except ImportError:  # Windows: nessun lock, uso sequenziale
            pass
        yield
    finally:
        f.close()


def load() -> dict:
    if not STATE.exists():
        sys.exit("Stato non inizializzato: esegui `python3 tools/state.py init --data-dir <percorso>`")
    return json.loads(STATE.read_text())


def save(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    os.replace(tmp, STATE)


def phase(st: dict, pid: str) -> dict:
    for p in st["phases"]:
        if p["id"] == pid:
            return p
    sys.exit(f"Fase sconosciuta: {pid}")


def report_path(p: dict) -> Path:
    return REPORTS / f"{p['id']}_{p['name']}.md"


def handoff_text(p: dict) -> str | None:
    rp = report_path(p)
    if not rp.exists():
        return None
    m = re.search(r"^## Handoff\s*\n(.*?)(?=^## |\Z)", rp.read_text(), re.S | re.M)
    return m.group(1).strip() if m else None


def satisfied(st: dict, pid: str) -> bool:
    return phase(st, pid)["status"] in ("done", "skipped")


def runnable(st: dict) -> list[dict]:
    out = [p for p in st["phases"] if p["status"] == "pending" and all(satisfied(st, d) for d in p["depends_on"])]
    return sorted(out, key=lambda p: (-p.get("priority", 0), p["id"]))


def append_decision(kind: str, pid: str, text: str) -> None:
    AG.mkdir(parents=True, exist_ok=True)
    if not DECISIONS.exists():
        DECISIONS.write_text("# Registro decisioni (append-only)\n\n| Quando | Tipo | Fase | Testo |\n|---|---|---|---|\n")
    with open(DECISIONS, "a") as f:
        f.write(f"| {now()} | {kind} | {pid or '-'} | {text.replace(chr(10), ' ').replace('|', '/')} |\n")


# ------------------------------------------------------------------ comandi
def cmd_init(a):
    with locked():
        if STATE.exists() and not a.force:
            sys.exit("Stato già presente (usa --force per ricominciare da zero: cancella lo stato, non i report).")
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["data_dir"] = str(Path(a.data_dir).expanduser().resolve()) if a.data_dir else None
        if a.primary_target:
            cfg["primary_target"] = a.primary_target
        phases = []
        for p in DEFAULT_PHASES:
            q = dict(p)
            q.update(status="pending", attempts=0, started=None, finished=None, summary=None, kind="planned",
                     approved=not q.get("requires_approval", False), report=f"reports/{p['id']}_{p['name']}.md")
            phases.append(q)
        save({"version": 1, "created": now(), "config": cfg, "phases": phases, "followups_used": 0,
              "review_rounds": 0, "frozen": False, "test_used": False})
    # regola di lettura negativa sulla cartella dei dati (gli agenti non devono aprire i file con lo strumento Read)
    if cfg["data_dir"]:
        sd = ROOT / ".claude"
        sd.mkdir(exist_ok=True)
        local = sd / "settings.local.json"
        cur = json.loads(local.read_text()) if local.exists() else {}
        deny = cur.setdefault("permissions", {}).setdefault("deny", [])
        rule = f"Read(/{cfg['data_dir']}/**)" if cfg["data_dir"].startswith("/") else f"Read({cfg['data_dir']}/**)"
        if rule not in deny:
            deny.append(rule)
        local.write_text(json.dumps(cur, indent=2))
        print(f"Aggiunta regola di blocco lettura: {rule}  (in .claude/settings.local.json)")
    append_decision("config", "-", f"pipeline inizializzata, data_dir={cfg['data_dir']}")
    print("Stato inizializzato:", STATE)


def cmd_status(a):
    st = load()
    if a.json:
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return
    print(f"{'ID':<9}{'FASE':<16}{'AGENTE':<20}{'STATO':<10}{'TENT':<5}{'DIPENDE DA'}")
    for p in st["phases"]:
        extra = " [serve approvazione]" if p.get("requires_approval") and not p.get("approved") else ""
        print(f"{p['id']:<9}{p['name']:<16}{p['agent']:<20}{p['status']:<10}{p['attempts']:<5}{','.join(p['depends_on']) or '-'}{extra}")


def cmd_next(a):
    st = load()
    r = runnable(st)
    if a.json:
        print(json.dumps([p["id"] for p in r]))
    else:
        print("\n".join(f"{p['id']} {p['name']} ({p['agent']})" for p in r) or "(nessuna fase eseguibile)")


def cmd_start(a):
    with locked():
        st = load()
        p = phase(st, a.id)
        p.update(status="running", started=now(), finished=None)
        p["attempts"] += 1
        save(st)


def cmd_done(a):
    with locked():
        st = load()
        p = phase(st, a.id)
        p.update(status="done", finished=now(), summary=a.summary, gate_pending=bool(p.get("gate")))
        save(st)


def cmd_fail(a):
    with locked():
        st = load()
        p = phase(st, a.id)
        p.update(status="failed", finished=now(), summary=a.reason)
        save(st)
    append_decision("issue", a.id, f"fase fallita: {a.reason}")


def cmd_skip(a):
    with locked():
        st = load()
        p = phase(st, a.id)
        p.update(status="skipped", finished=now(), summary=a.reason)
        save(st)
    append_decision("decision", a.id, f"fase saltata: {a.reason}")


def cmd_reset(a):
    with locked():
        st = load()
        p = phase(st, a.id)
        p.update(status="pending", finished=None)
        if a.attempts:
            p["attempts"] = 0
        save(st)


def cmd_recover(a):
    """Le fasi 'running' dopo un crash/interruzione tornano 'pending' (il lavoro parziale resta su disco)."""
    with locked():
        st = load()
        rec = []
        for p in st["phases"]:
            if p["status"] == "running":
                p["status"] = "pending"
                p["interrupted"] = now()
                rec.append(p["id"])
        save(st)
    for pid in rec:
        append_decision("issue", pid, "fase interrotta: riparte dai file parziali in results/ e dal file .progress.md")
    print("recuperate:", ",".join(rec) or "nessuna")


def cmd_bump_review(a):
    with locked():
        st = load()
        if st["review_rounds"] >= st["config"]["max_review_rounds"]:
            sys.exit(f"Round di revisione esauriti ({st['config']['max_review_rounds']}): congela solo ciò che il revisore ha ammesso.")
        st["review_rounds"] += 1
        save(st)
    append_decision("decision", "P6", f"avviato round di revisione {st['review_rounds']}")
    print("review_rounds =", st["review_rounds"])


def cmd_gate_done(a):
    with locked():
        st = load()
        phase(st, a.id)["gate_pending"] = False
        save(st)


def cmd_approve(a):
    with locked():
        st = load()
        phase(st, a.id)["approved"] = True
        save(st)
    append_decision("decision", a.id, "approvazione umana per l'esecuzione")


def cmd_add_phase(a):
    with locked():
        st = load()
        cfg = st["config"]
        if not a.revises and st["followups_used"] >= cfg["max_followups"]:
            sys.exit(f"Limite di fasi di follow-up raggiunto ({cfg['max_followups']}). Segnala in DECISIONS.md e procedi.")
        if any(p["id"] == a.id for p in st["phases"]):
            sys.exit(f"Esiste già una fase {a.id}")
        deps = [d for d in (a.depends_on or "").split(",") if d]
        for d in deps:
            phase(st, d)
        st["phases"].append(dict(id=a.id, name=a.name, agent=a.agent, depends_on=deps, gate=a.gate, priority=a.priority,
                                 goal=a.goal, status="pending", attempts=0, started=None, finished=None, summary=None,
                                 kind="followup", revises=a.revises, approved=True, report=f"reports/{a.id}_{a.name}.md"))
        if not a.revises:
            st["followups_used"] += 1
        # il revisore e le fasi finali devono attendere anche il follow-up appena aggiunto
        for pid in (a.blocks or "").split(","):
            if pid:
                q = phase(st, pid)
                if a.id not in q["depends_on"]:
                    q["depends_on"].append(a.id)
                if q["status"] in ("done", "skipped") and a.reopen:
                    q.update(status="pending", finished=None)
        save(st)
    append_decision("decision", a.id, f"aggiunta fase di follow-up {a.id} ({a.name}): {a.goal}")


def cmd_set_config(a):
    with locked():
        st = load()
        try:
            val = json.loads(a.value)
        except json.JSONDecodeError:
            val = a.value
        cur = st["config"]
        keys = a.key.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = val
        save(st)
    append_decision("config", "-", f"{a.key} = {a.value}")


def cmd_set_priority(a):
    with locked():
        st = load()
        phase(st, a.id)["priority"] = a.priority
        save(st)


def cmd_note(a):
    append_decision(a.kind, a.phase, a.text)


def cmd_log_exp(a):
    REG.parent.mkdir(parents=True, exist_ok=True)
    new = not REG.exists()
    with locked():
        n = 1
        if not new:
            with open(REG) as f:
                n = sum(1 for _ in f)
        with open(REG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "exp_id", "phase", "target", "uses_val", "n_configs", "metric_val", "description"])
            w.writerow([now(), f"E{n:04d}", a.phase, a.target or "", int(a.uses_val), a.n_configs, a.metric_val or "", a.desc])
    print(f"E{n:04d}")


def cmd_handoff(a):
    st = load()
    p = phase(st, a.id)
    h = handoff_text(p)
    print(h if h else f"(nessun Handoff trovato per {a.id}: report mancante o senza sezione)")


def cmd_freeze(a):
    """Segna la lista dei candidati come congelata (FROZEN.json scritto dall'orchestratore)."""
    fp = AG / "FROZEN.json"
    if not fp.exists():
        sys.exit("agentic/FROZEN.json non esiste")
    fz = json.loads(fp.read_text())
    for c in fz.get("candidates", []):
        d = ROOT / c["dir"]
        if not (d / "candidate.py").exists() or not (d / "candidate.json").exists():
            sys.exit(f"Candidato senza candidate.py/candidate.json: {c['dir']}")
    with locked():
        st = load()
        st["frozen"] = True
        save(st)
    append_decision("decision", "P6", f"congelati {len(fz.get('candidates', []))} candidati per la valutazione sul test")
    print("congelati:", len(fz.get("candidates", [])))


def registry_summary() -> str:
    if not REG.exists():
        return "nessuna configurazione registrata"
    rows = list(csv.DictReader(open(REG)))
    by: dict[str, int] = {}
    for r in rows:
        by[r["phase"]] = by.get(r["phase"], 0) + int(r["n_configs"] or 1)
    return ", ".join(f"{k}: {v}" for k, v in sorted(by.items())) + f"  (totale {sum(by.values())}, esperimenti {len(rows)})"


def cmd_brief(a):
    st = load()
    cfg = st["config"]
    print(f"# BRIEF PIPELINE  ({now()})\n")
    print(f"Target primario: {cfg['primary_target']} | data_dir: {cfg['data_dir']}")
    print(f"tabular_signal={cfg.get('tabular_signal')} | cluster_targets={cfg.get('cluster_targets')} | "
          f"clusters_mode={cfg.get('clusters_mode')} | follow-up usati {st['followups_used']}/{cfg['max_followups']} | "
          f"revisioni {st['review_rounds']}/{cfg['max_review_rounds']} | "
          f"test_usato={st['test_used']} | candidati_congelati={st['frozen']}")
    print(f"Configurazioni provate (per multiplicità): {registry_summary()}\n")
    cmd_status(argparse.Namespace(json=False))
    interrupted = [p["id"] for p in st["phases"] if p["status"] == "running"]
    if interrupted:
        print(f"\nATTENZIONE: fasi 'running' (probabile interruzione): {','.join(interrupted)} -> `python3 tools/state.py recover`")
    owed = [p["id"] for p in st["phases"] if p["status"] == "done" and p.get("gate_pending")]
    if owed:
        print(f"\nGATE DOVUTI (fase completata, decisioni dell'orchestratore non ancora prese): {','.join(owed)}")
    r = runnable(st)
    paused = [p["id"] for p in r if p.get("requires_approval") and not p.get("approved")]
    print("\nProssime eseguibili:", ", ".join(f"{p['id']}({p['agent']})" for p in r) or "nessuna",
          "| in attesa di approvazione umana:" if paused else "", ",".join(paused))
    failed = [p for p in st["phases"] if p["status"] == "failed"]
    for p in failed:
        print(f"FALLITA {p['id']}: {p['summary']}")
    print("\n## Handoff delle fasi completate")
    for p in st["phases"]:
        if p["status"] == "done":
            h = handoff_text(p)
            lines = (h or "(report senza Handoff)").splitlines()[: a.max_lines]
            print(f"\n### {p['id']} {p['name']}\n" + "\n".join(lines))
        elif p["status"] == "skipped":
            print(f"\n### {p['id']} {p['name']}: SALTATA ({p['summary']})")
    if DECISIONS.exists():
        tail = DECISIONS.read_text().splitlines()[-a.decisions:]
        print("\n## Ultime decisioni\n" + "\n".join(tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sp.add_parser(name)
        s.set_defaults(fn=fn)
        return s

    s = add("init", cmd_init); s.add_argument("--data-dir"); s.add_argument("--primary-target"); s.add_argument("--force", action="store_true")
    s = add("status", cmd_status); s.add_argument("--json", action="store_true")
    s = add("next", cmd_next); s.add_argument("--json", action="store_true")
    s = add("start", cmd_start); s.add_argument("id")
    s = add("done", cmd_done); s.add_argument("id"); s.add_argument("--summary", default="")
    s = add("fail", cmd_fail); s.add_argument("id"); s.add_argument("--reason", required=True)
    s = add("skip", cmd_skip); s.add_argument("id"); s.add_argument("--reason", required=True)
    s = add("reset", cmd_reset); s.add_argument("id"); s.add_argument("--attempts", action="store_true")
    add("recover", cmd_recover)
    s = add("approve", cmd_approve); s.add_argument("id")
    s = add("gate-done", cmd_gate_done); s.add_argument("id")
    add("bump-review", cmd_bump_review)
    s = add("add-phase", cmd_add_phase)
    s.add_argument("--id", required=True); s.add_argument("--name", required=True); s.add_argument("--agent", required=True)
    s.add_argument("--goal", required=True); s.add_argument("--depends-on", default=""); s.add_argument("--priority", type=int, default=0)
    s.add_argument("--gate", action="store_true"); s.add_argument("--revises"); s.add_argument("--blocks", default="")
    s.add_argument("--reopen", action="store_true", help="riapre le fasi in --blocks se già completate")
    s = add("set-config", cmd_set_config); s.add_argument("key"); s.add_argument("value")
    s = add("set-priority", cmd_set_priority); s.add_argument("id"); s.add_argument("priority", type=int)
    s = add("note", cmd_note); s.add_argument("--kind", default="decision", choices=["decision", "issue", "assumption", "config"])
    s.add_argument("--phase", default="-"); s.add_argument("--text", required=True)
    s = add("log-exp", cmd_log_exp); s.add_argument("--phase", required=True); s.add_argument("--desc", required=True)
    s.add_argument("--target"); s.add_argument("--uses-val", action="store_true"); s.add_argument("--n-configs", type=int, default=1)
    s.add_argument("--metric-val")
    s = add("handoff", cmd_handoff); s.add_argument("id")
    add("freeze", cmd_freeze)
    s = add("brief", cmd_brief); s.add_argument("--max-lines", type=int, default=25); s.add_argument("--decisions", type=int, default=8)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
