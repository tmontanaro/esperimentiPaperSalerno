#!/usr/bin/env python3
"""Stato della PIPELINE 2 (fasi Q0-Q10). Stessi comandi di tools/state.py, ma su file separati:
agentic/pipeline2_state.json, agentic/DECISIONS2.md, agentic/RUNLOG2.md, agentic/FROZEN2.json.
Il registro degli esperimenti results/registry.csv è CONDIVISO con la pipeline 1 (così il conteggio delle configurazioni provate è onesto).

  python3 tools/state2.py init            # prende data_dir e soglie dalla pipeline 1
  python3 tools/state2.py brief | next | handoff Q1 | note ... | log-exp ... | freeze | approve Q8 | recover | reset Q1 --attempts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import state as S  # noqa: E402
import phases2 as PH  # noqa: E402

S.STATE = S.AG / "pipeline2_state.json"
S.LOCK = S.AG / ".state2.lock"
S.DECISIONS = S.AG / "DECISIONS2.md"
S.DEFAULT_PHASES = PH.PHASES
FROZEN2 = S.AG / "FROZEN2.json"
P1_STATE = S.AG / "pipeline_state.json"
PROTECTED = ["agentic/pipeline2_state.json", "agentic/PLAYBOOK2.md", "agentic/RUNLOG2.md", "agentic/.state2.lock"]


def init2(a):
    import copy
    import json as _j
    with S.locked():
        if S.STATE.exists() and not a.force:
            sys.exit("Stato della pipeline 2 già presente (usa --force per ricominciare: cancella lo stato, non i report).")
        cfg = copy.deepcopy(PH.CONFIG2)
        p1 = _j.loads(P1_STATE.read_text()) if P1_STATE.exists() else None
        if p1:
            cfg["data_dir"] = p1["config"].get("data_dir")
            cfg["thresholds"] = p1["config"].get("thresholds", cfg["thresholds"])
            cfg["primary_target"] = p1["config"].get("primary_target", cfg["primary_target"])
            done = [p["id"] for p in p1["phases"] if p["status"] in ("done", "skipped")]
            if "P8" not in done:
                print("ATTENZIONE: la pipeline 1 non risulta completa (P8 non finita). La pipeline 2 si basa sui suoi risultati.", file=sys.stderr)
        elif not a.data_dir:
            sys.exit("Pipeline 1 non trovata (agentic/pipeline_state.json): indica --data-dir.")
        if a.data_dir:
            cfg["data_dir"] = str(Path(a.data_dir).expanduser().resolve())
        phases = []
        for p in PH.PHASES:
            q = dict(p)
            q.update(status="pending", attempts=0, started=None, finished=None, summary=None, kind="planned",
                     approved=not q.get("requires_approval", False), report=f"reports/{p['id']}_{p['name']}.md")
            phases.append(q)
        S.save({"version": 1, "pipeline": 2, "created": S.now(), "config": cfg, "phases": phases, "followups_used": 0,
                "review_rounds": 0, "frozen": False, "test_used": False})
    # permessi: blocco lettura dati + protezione dei file della pipeline 2
    sd = ROOT / ".claude"
    sd.mkdir(exist_ok=True)
    local = sd / "settings.local.json"
    cur = _j.loads(local.read_text()) if local.exists() else {}
    deny = cur.setdefault("permissions", {}).setdefault("deny", [])
    rules = [f"Edit(./{f})" for f in PROTECTED]
    if cfg["data_dir"]:
        rules.append(f"Read(/{cfg['data_dir']}/**)" if cfg["data_dir"].startswith("/") else f"Read({cfg['data_dir']}/**)")
    for r in rules:
        if r not in deny:
            deny.append(r)
    local.write_text(_j.dumps(cur, indent=2))
    S.append_decision("config", "-", f"pipeline 2 inizializzata, data_dir={cfg['data_dir']}")
    print("Stato pipeline 2 inizializzato:", S.STATE)
    print("Regole di permesso aggiunte in .claude/settings.local.json:", len(rules))


def _sha(d: Path) -> dict:
    out = {}
    for f in sorted(d.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            out[str(f.relative_to(d))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def freeze2(a):
    """Congela i candidati del secondo test: controlla il contratto e registra le impronte (sha256) di ogni file."""
    if not FROZEN2.exists():
        sys.exit("agentic/FROZEN2.json non esiste")
    fz = json.loads(FROZEN2.read_text())
    cands = fz.get("candidates", [])
    st = S.load()
    mx = st["config"].get("max_candidates_second_test", 6)
    if len(cands) > mx:
        sys.exit(f"Troppi candidati ({len(cands)} > {mx}): il numero di confronti sul test va tenuto basso.")
    for c in cands:
        d = ROOT / c["dir"]
        if not (d / "candidate.py").exists() or not (d / "candidate.json").exists():
            sys.exit(f"Candidato senza candidate.py/candidate.json: {c['dir']}")
        c["sha256"] = _sha(d)
        wd = d / "weights_ref.json"
        # i pesi stanno nella cartella indicata da candidate.py (WEIGHTS): ne congeliamo l'impronta
        src = (d / "candidate.py").read_text()
        import re
        m = re.search(r'WEIGHTS = ROOT / "([^"]+)"', src)
        if m and (ROOT / m.group(1)).exists():
            c["weights_dir"] = m.group(1)
            c["weights_sha256"] = _sha(ROOT / m.group(1))
    fz["frozen_at"] = S.now()
    FROZEN2.write_text(json.dumps(fz, indent=2))
    with S.locked():
        st = S.load(); st["frozen"] = True; S.save(st)
    S.append_decision("decision", "Q6", f"congelati {len(cands)} candidati per il secondo test (impronte sha256 registrate)")
    print("congelati:", len(cands))


def brief2(a):
    st = S.load()
    cfg = st["config"]
    print(f"# BRIEF PIPELINE 2  ({S.now()})\n")
    print(f"Target primario: {cfg['primary_target']} | data_dir: {cfg['data_dir']}")
    print(f"fold={cfg.get('n_folds')} seed={cfg.get('seeds')} | candidati_congelati={st['frozen']} | secondo_test_usato={st['test_used']} | "
          f"max candidati secondo test={cfg.get('max_candidates_second_test')}")
    print(f"Configurazioni provate (per multiplicità, pipeline 1 + 2): {S.registry_summary()}\n")
    S.cmd_status(argparse.Namespace(json=False))
    interrupted = [p["id"] for p in st["phases"] if p["status"] == "running"]
    if interrupted:
        print(f"\nATTENZIONE: fasi 'running' (probabile interruzione): {','.join(interrupted)} -> `python3 tools/state2.py recover`")
    r = S.runnable(st)
    paused = [p["id"] for p in r if p.get("requires_approval") and not p.get("approved")]
    print("\nProssime eseguibili:", ", ".join(f"{p['id']}({p['agent']})" for p in r) or "nessuna",
          "| in attesa di approvazione umana:" if paused else "", ",".join(paused))
    for p in [p for p in st["phases"] if p["status"] == "failed"]:
        print(f"FALLITA {p['id']}: {p['summary']}")
    print("\n## Handoff delle fasi completate")
    for p in st["phases"]:
        if p["status"] == "done":
            h = S.handoff_text(p)
            print(f"\n### {p['id']} {p['name']}\n" + "\n".join((h or "(report senza Handoff)").splitlines()[: a.max_lines]))
        elif p["status"] == "skipped":
            print(f"\n### {p['id']} {p['name']}: SALTATA ({p['summary']})")
    if S.DECISIONS.exists():
        print("\n## Ultime decisioni\n" + "\n".join(S.DECISIONS.read_text().splitlines()[-a.decisions:]))
    print("\nRicorda: in questa pipeline usa SEMPRE tools/state2.py (non tools/state.py).")


def status2(a):
    st = S.load()
    if a.json:
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return
    print(f"{'ID':<5}{'FASE':<24}{'AGENTE':<24}{'STATO':<10}{'TENT':<5}{'DIPENDE DA'}")
    for p in st["phases"]:
        extra = " [serve approvazione]" if p.get("requires_approval") and not p.get("approved") else ""
        print(f"{p['id']:<5}{p['name']:<24}{p['agent']:<24}{p['status']:<10}{p['attempts']:<5}{','.join(p['depends_on']) or '-'}{extra}")


S.cmd_status = status2
S.cmd_init = init2
S.cmd_freeze = freeze2
S.cmd_brief = brief2

if __name__ == "__main__":
    S.main()
