"""Assembla reports/Q9_synthesis2.md e reports/REPORT_COMPLESSIVO.md dal testo results/Q9_body.md. Nessun dato di paziente."""
raw = open('results/Q9_body.md').read().replace("un'aggiustamento", "un aggiustamento")
body = ("\n" + raw).replace("\n## ", "\n### ").lstrip("\n")
idx = """# Report complessivo EchoNext: ECG e cardiopatia strutturale
Data: 2026-11-09. Copia autonoma del report Q9 (pipeline 1 e 2). Solo numeri aggregati; il test è chiuso.

Indice: In breve · A. Obiettivo · B. Risultati · C. Prove fatte · D. Fattibile e non · E. Cosa si può ancora fare · F. Quanto fidarsi · Glossario · Come riprodurre

"""
open('reports/REPORT_COMPLESSIVO.md', 'w').write(idx + raw)
head = """# Report Q9 · synthesis2
- Agente: synthesizer2
- Data: 2026-11-09 18:00
- Stato: OK

## Obiettivo
Spiegare in parole povere a colleghi non esperti di ML l'intero lavoro (pipeline 1 e 2): obiettivo, prove, risultati, cosa è fattibile e cosa no. SHD non coincide con scompenso cardiaco. La copia autonoma è `reports/REPORT_COMPLESSIVO.md`.

## Input letti
Brief e Handoff di `tools/state2.py` (Q0-Q8); `reports/Q8_final-test2.md`, `reports/FINAL_REPORT.md` (P8), Handoff di P0 e P1, `reports/final_test_table.md`, `reports/final_test2_table.md`, `reports/Q9_synthesis2.progress.md`. Il registro non è stato riletto riga per riga: i conteggi vengono dal brief (725 configurazioni, 40 esperimenti). I report P2-P7 e Q1-Q7 sono stati usati attraverso i loro Handoff e P8, non riletti per intero. Il test non è stato aperto.

## Cosa è stato fatto
Solo sintesi di documenti esistenti, nessun calcolo, nessuno smoke test. Il primo tentativo era stato interrotto prima di scrivere il report; questo è il riavvio. Script di assemblaggio: `experiments/Q9/build_q9.py`.

## Risultati
"""
tail = """
## Decisioni e motivazioni
Calcoli non rifatti: i numeri sono copiati dai report. Dove le fonti discordano sono riportate entrambe (sezione F). Per il test la popolazione principale è clean, come deciso in Q8.

## Configurazioni provate
0 nuove configurazioni, 0 con uso della validation. Totale ereditato dal brief: 725 (pipeline 1: 511, pipeline 2: 214), 40 esperimenti.

## Artefatti
`reports/Q9_synthesis2.md`, `reports/REPORT_COMPLESSIVO.md`, `reports/Q9_synthesis2.progress.md`, `results/Q9_body.md` (testo di lavoro), `experiments/Q9/build_q9.py`.

## Problemi aperti
- I numeri di pipeline 1 per P2-P7 sono di seconda mano (da P8 e dagli Handoff), non riletti dai report originali.
- La sezione E contiene proposte, non risultati.
- Nessuna validazione esterna, test letto due volte, righe senza LVEF anomale.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- Report: `reports/Q9_synthesis2.md` e copia `reports/REPORT_COMPLESSIVO.md`.
- VERDETTO_FATTIBILITA: IN PARTE (CNN fattibile per SHD, LVEF ≤ 45%, VD, stenosi aortica; no per sottogruppi, QRS, LVEF numerica precisa).
- Test clean, IC95% per paziente: SHD 0.824 [0.813-0.836]; LVEF≤45% 0.897 [0.886-0.908]; VD 0.889 [0.873-0.904]; stenosi 0.832 [0.811-0.857]; SHD 3 deriv. 0.818 [0.808-0.829].
- Tabulare SHD test 0.762 [0.750-0.774]; CNN vs tabulare +0.075 [+0.063, +0.086].
- Sottogruppi/cluster credibili: 0 (1950 regole + 4 cluster in P; 15 cluster in Q4). QRS: +0.0001 [-0.0005, +0.0006].
- 725 configurazioni; test letto due volte (P7, Q8).
RACCOMANDAZIONE:
- Q10: costruire il notebook dai numeri di questo report; non riaprire il test.
"""
open('reports/Q9_synthesis2.md', 'w').write(head + body + tail)
