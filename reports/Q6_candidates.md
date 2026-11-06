# Report Q6 · candidates
- Agente: candidate-freezer
- Data: 2026-11-06 20:50
- Stato: OK

## Obiettivo
Scegliere, con regole scritte PRIMA di guardare il test, i candidati del secondo test; addestrarli sull'intero pool di sviluppo (train + validation, popolazione «clean») e congelarli. Il test non è stato aperto. Ricordiamo che SHD (cardiopatia strutturale) non coincide con lo scompenso cardiaco.

## Input letti
- Handoff di Q2, Q3, Q4, Q5 e di Q1 (via `state2.py brief`): CONFIG_FINALE di Q1 (12 epoche, lr 0.002, reg_weight 0.5, batch 128, 12 derivazioni, clean); TARGETS_GOOD = SHD, LVEF ≤ 45%, disfunzione sistolica VD, stenosi aortica.
- `results/Q1/main/metrics_clean.csv`: AUROC CV clean e numero di positivi per etichetta.
- Q2: INSIEME_MINIMO_SUFFICIENTE = 3 derivazioni (I, II + una toracica; `LEAD_ORDER_OK` = yes, le toraciche non sono nominate), AUROC SHD 0.802 [0.795-0.809] (seed 0).
- Q3, Q4, Q5: nessun candidato nuovo (ipotesi QRS non supportata; 0 cluster credibili su 15; calibrazione solo diagnostica).

## Cosa è stato fatto
1. Scritta `results/Q6/preregistration.md` (regole e criteri di conferma) prima di qualunque addestramento.
2. `tools/cnn_cv.py final-fit` con 12 derivazioni (`results/Q6/final_all`, 3 seed 0,1,2, 12 epoche; ~11 s/epoca; 2 lanci per il limite di 7 minuti) e con 3 derivazioni (`results/Q6/final_I_II_V1`, ~6 s/epoca). Entrambi COMPLETI. Addestrati sull'intero pool clean, quindi non esiste un valore di validation indipendente per questi pesi.
3. `make-candidate` per 5 candidati. Un primo tentativo di C3 con un nome di etichetta errato (`rv_systolic_dysfunction_flag`) è stato rigenerato con quello corretto (`..._moderate_or_greater_flag`) prima del congelamento.
4. Verifica del contratto: `experiments/Q6/check_contract.py` chiama `predict(df, lib)` su 40 righe di validation (solo forma, indice, intervallo [0,1]): 5 su 5 OK. Nessuno smoke test con `debug_n`, perché si usa lo stesso codice già validato in Q1/Q2.
5. Scritto `agentic/FROZEN2.json` ed eseguito `python3 tools/state2.py freeze` (5 candidati congelati, impronte sha256 registrate, 2026-11-06 20:45:03).

## Risultati
Nessun valore di test. Gli AUROC sono di cross-validation per paziente (5 fold, ensemble di 3 seed, popolazione clean, Q1/Q2), da usare come riferimento per il calo al test.

| Candidato | Uscita | Derivazioni | AUROC CV clean [IC95% per paziente] | n clean | positivi | Comparatore tabulare |
|---|---|---|---|---|---|---|
| C1_shd_all | SHD | 12 | 0.815 [0.809-0.822] | 70130 | 39948 | P1 global_best |
| C2_lvef45_all | LVEF ≤ 45% | 12 | 0.869 [0.862-0.876] | 70130 | 17828 | P3 lvef_lte45_global |
| C3_rv_all | disfunzione sistolica VD | 12 | 0.844 [0.834-0.853] | 70130 | 9965 | nessuno |
| C4_as_all | stenosi aortica | 12 | 0.811 [0.794-0.826] | 70130 | 3171 | nessuno (in CV il tabulare 0.841 [0.825-0.857] era migliore: differenza CNN-tab −0.030 [−0.045, −0.014]) |
| C5_shd_I_II_V1 | SHD | 3 | 0.802 [0.795-0.809] (solo seed 0) | 70130 | 39948 | P1 global_best |

Criteri di conferma al test (fissati prima): «confermato» = AUROC test con IC inferiore ≥ 0.75 e calo dal CV ≤ 0.05; «buona» = AUROC ≥ 0.80 con IC inferiore ≥ 0.75. Candidati: 5 su un massimo di 6.

## Decisioni e motivazioni
- C1 e C2 sempre; C3 e C4 perché in TARGETS_GOOD con ≥ 100 positivi (9965 e 3171), in ordine di AUROC; C5 perché Q2 ha trovato un set minimo di 3 derivazioni (perdita −0.006 [−0.009, −0.003] su SHD, entro la soglia 0.01 definita in Q2: «sufficiente» non significa «senza perdita»).
- Scartato il set `eight` (nessuna perdita): non richiesto dalle regole, il candidato in più è uno solo.
- Nessun candidato da cluster o sottogruppi: Q4 ha 0 cluster credibili, e gli strati sono diagnostici e correlati.
- Il riferimento CV di C5 è un solo seed, mentre il candidato è un ensemble di 3 seed: il confronto con il test sarà leggermente conservativo.
- Con la stenosi aortica ci si aspetta che la CNN non superi il tabulare.

## Configurazioni provate
2 configurazioni (due final-fit: 12 derivazioni e 3 derivazioni), registrate con `state2.py log-exp` (E0039, E0040, `--uses-val`; il pool di sviluppo comprende il validation). Nessuna scelta di iperparametri in questa fase: usano la CONFIG_FINALE di Q1. La scelta di C5 deriva dalle 7 configurazioni di Q2, già contate lì.

## Artefatti
- `results/Q6/preregistration.md`
- `results/Q6/final_all/`, `results/Q6/final_I_II_V1/` (pesi, 3 seed)
- `results/Q6/candidates/{C1_shd_all,C2_lvef45_all,C3_rv_all,C4_as_all,C5_shd_I_II_V1}/`
- `experiments/Q6/check_contract.py`
- `agentic/FROZEN2.json` (impronte), `reports/Q6_candidates.progress.md`

## Problemi aperti
- I pesi sono addestrati su tutto il pool incluso il validation: nessuna stima indipendente prima del test; vale solo il CV (che è per paziente).
- C5 ha un riferimento CV da un solo seed (0.802).
- Il calo CV→test può contenere un effetto di popolazione (prevalenza SHD 0.575 nel train e 0.484 nel validation, Q5): non è calibrazione ma può spostare le metriche non discriminative.
- Le righe senza `lvef_value` (circa 9%) sono anomale e escluse dal training: limite dichiarato, non chiarito.
- Il tabulare non ha comparatore per VD e stenosi aortica: la differenza dal tabulare non sarà valutabile al test per questi.
- Dopo il congelamento nulla può essere cambiato.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- CANDIDATI: [C1_shd_all: SHD, 12 deriv.; C2_lvef45_all: LVEF≤45, 12; C3_rv_all: VD, 12; C4_as_all: stenosi aortica, 12; C5_shd_I_II_V1: SHD, 3 deriv.]
- N_CANDIDATI: 5 (massimo 6); congelati il 2026-11-06 20:45:03
- REGOLE_PRESPECIFICATE: results/Q6/preregistration.md
- CRITERI_CONFERMA: confermato = AUROC test IC inf ≥ 0.75 e calo da CV ≤ 0.05; buona = AUROC ≥ 0.80 con IC inf ≥ 0.75
- AUROC CV clean di riferimento: SHD 0.815, LVEF≤45 0.869, VD 0.844, stenosi 0.811, SHD 3 deriv. 0.802 (seed 0)
- Configurazioni: 2 final-fit registrati (E0039, E0040)
RACCOMANDAZIONE:
- Q7: verificare le impronte sha256 in agentic/FROZEN2.json senza modificarle e che le regole siano state scritte prima del test.
- Q8: valutare i 5 candidati senza cambiare soglie; per C4 attendersi nessun vantaggio sul tabulare.
