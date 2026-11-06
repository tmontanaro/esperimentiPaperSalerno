#!/usr/bin/env python3
"""Driver della PIPELINE 2 (fasi Q0-Q10): stesse opzioni e stessi codici di uscita di tools/run_pipeline.py
(0 finita · 2 fallita/bloccata · 3 cartella non approvata · 4 claude non trovato · 10 attende approvazione · 75 limite d'uso),
ma con stato in agentic/pipeline2_state.json, log in agentic/RUNLOG2.md e istruzioni in agentic/PLAYBOOK2.md.

  python3 tools/state2.py init
  python3 tools/run_pipeline2.py --max-budget-usd 3 --timeout-min 120
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import state2  # noqa: E402,F401  (reindirizza lo stato su pipeline2_state.json)
import run_pipeline as R  # noqa: E402

R.RUNLOG = ROOT / "agentic" / "RUNLOG2.md"
R.BACKUPS = ROOT / "agentic" / "backups2"


def _adapt(t: str) -> str:
    return (t.replace("tools/state.py", "tools/state2.py").replace("agentic/PLAYBOOK.md", "agentic/PLAYBOOK2.md")
             .replace("pipeline EchoNext", "pipeline 2 EchoNext (fasi Q0-Q10, estensione della pipeline 1)"))


R.PHASE_PROMPT = _adapt(R.PHASE_PROMPT) + ("\nIn questa pipeline valgono le regole di CLAUDE.md con queste sostituzioni: tools/state2.py al posto di tools/state.py; "
                                            "il test lo apre solo tools/final_eval2.py (fase Q8). Per i lavori lunghi usa --time-budget-min e rilancia lo stesso comando (vedi PLAYBOOK2).")
R.GATE_PROMPT = _adapt(R.GATE_PROMPT)
R.FINAL_MSG = "Pipeline 2 completata. Report complessivo: reports/Q9_synthesis2.md (copia: reports/REPORT_COMPLESSIVO.md); notebook: reports/Q10_notebook/."
R.APPROVAL_MSG = ("Controlla agentic/FROZEN2.json (candidati congelati) e il verdetto di reports/Q7_review2.md; poi:  "
                  "python3 tools/state2.py approve {id}  e rilancia  python3 tools/run_pipeline2.py")
R.RESET_HINT = "Dopo aver risolto: python3 tools/state2.py reset <ID> --attempts   e rilancia python3 tools/run_pipeline2.py"

if __name__ == "__main__":
    try:
        sys.exit(R.main())
    except R.Untrusted:
        print("\n" + R.UNTRUSTED_MSG)
        sys.exit(3)
