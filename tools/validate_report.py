#!/usr/bin/env python3
"""Valida la struttura di un report di fase. Uso: validate_report.py reports/P1_baseline.md  (exit 0 = valido)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED = ["Obiettivo", "Input letti", "Cosa è stato fatto", "Risultati", "Decisioni e motivazioni",
            "Configurazioni provate", "Artefatti", "Problemi aperti", "Handoff"]
MAX_HANDOFF_LINES = 40


def check(path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return [f"file mancante: {p}"]
    txt = p.read_text()
    errs = []
    sections = {m.group(1).strip(): m.group(2) for m in re.finditer(r"^## (.+?)\s*\n(.*?)(?=^## |\Z)", txt, re.S | re.M)}
    for r in REQUIRED:
        if r not in sections:
            errs.append(f"sezione mancante: '## {r}'")
        elif len(sections[r].strip()) < 10:
            errs.append(f"sezione vuota o troppo breve: '## {r}'")
    if "Handoff" in sections:
        h = sections["Handoff"]
        lines = [l for l in h.splitlines() if l.strip()]
        if len(lines) > MAX_HANDOFF_LINES:
            errs.append(f"Handoff troppo lungo ({len(lines)} righe > {MAX_HANDOFF_LINES})")
        if not re.search(r"^STATUS:\s*(OK|PARZIALE|FALLITO)\s*$", h, re.M):
            errs.append("Handoff senza riga 'STATUS: OK|PARZIALE|FALLITO'")
        if not re.search(r"^RISULTATI_CHIAVE:", h, re.M):
            errs.append("Handoff senza riga 'RISULTATI_CHIAVE:'")
        if not re.search(r"^RACCOMANDAZIONE:", h, re.M):
            errs.append("Handoff senza riga 'RACCOMANDAZIONE:'")
    if "Risultati" in sections and not re.search(r"\d", sections["Risultati"]):
        errs.append("'## Risultati' non contiene numeri")
    if "Configurazioni provate" in sections and not re.search(r"\d", sections["Configurazioni provate"]):
        errs.append("'## Configurazioni provate' non indica quante configurazioni sono state provate")
    return errs


def status_of(path) -> str | None:
    m = re.search(r"^STATUS:\s*(OK|PARZIALE|FALLITO)\s*$", Path(path).read_text(), re.M)
    return m.group(1) if m else None


if __name__ == "__main__":
    e = check(sys.argv[1])
    if e:
        print("REPORT NON VALIDO:\n- " + "\n- ".join(e))
        sys.exit(1)
    print("report valido; STATUS =", status_of(sys.argv[1]))
