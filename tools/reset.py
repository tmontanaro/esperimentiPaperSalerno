#!/usr/bin/env python3
"""Azzera lo stato generato dalla pipeline (stato, report, risultati, esperimenti, log). NON tocca dati, lib, tools, agenti.
Uso: python3 tools/reset.py --yes   (senza --yes mostra solo cosa cancellerebbe)"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
targets = [ROOT / "agentic" / n for n in ("pipeline_state.json", "DECISIONS.md", "RUNLOG.md", "FROZEN.json", ".state.lock")]
targets += [ROOT / "agentic" / "backups", ROOT / ".claude" / "settings.local.json"]
for d in ("reports", "results", "experiments"):
    targets += [p for p in (ROOT / d).glob("*") if p.name != ".gitkeep"]
existing = [t for t in targets if t.exists()]
for t in existing:
    print(("cancello " if "--yes" in sys.argv else "cancellerei ") + str(t.relative_to(ROOT)))
if "--yes" in sys.argv:
    for t in existing:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
    print("fatto.")
elif existing:
    print("\nNulla è stato cancellato: rilancia con --yes per confermare.")
else:
    print("niente da cancellare.")
