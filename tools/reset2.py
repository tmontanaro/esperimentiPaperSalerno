#!/usr/bin/env python3
"""Azzera SOLO ciò che ha generato la pipeline 2 (stato, log, FROZEN2, report/risultati/esperimenti con prefisso Q). NON tocca la pipeline 1.
Uso: python3 tools/reset2.py --yes   (senza --yes mostra solo cosa cancellerebbe)
Nota: results/registry.csv (condiviso) NON viene toccato."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
targets = [ROOT / "agentic" / n for n in ("pipeline2_state.json", "DECISIONS2.md", "RUNLOG2.md", "FROZEN2.json", ".state2.lock", "backups2")]
for d in ("reports", "results", "experiments"):
    targets += [p for p in (ROOT / d).glob("Q*") if p.name[1:2].isdigit()]
targets += [ROOT / "results" / "final2", ROOT / "reports" / "final_test2_table.md", ROOT / "reports" / "REPORT_COMPLESSIVO.md"]
existing = [t for t in targets if t.exists()]
for t in existing:
    print(("cancello " if "--yes" in sys.argv else "cancellerei ") + str(t.relative_to(ROOT)))
if "--yes" in sys.argv:
    for t in existing:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
    print("fatto. (Le regole in .claude/settings.local.json restano: sono innocue.)")
elif existing:
    print("\nNulla è stato cancellato: rilancia con --yes per confermare.")
else:
    print("niente da cancellare.")
