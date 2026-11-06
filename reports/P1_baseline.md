# Report P1 · baseline
- Agente: baseline-modeler
- Data: 2026-11-06 14:30
- Stato: OK

## Obiettivo
Stabilire quanto segnale c'è nelle sole misure ECG tabulari (7 feature) per predire `shd_moderate_or_greater_flag`, confrontandole con un riferimento minimo (età + sesso). Nota: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco. Da questo risultato dipendono le decisioni delle fasi successive.

## Input letti
- `python3 tools/state.py handoff P0`: N_TRAIN 2127, N_VAL 252, prevalenza train 0,646 (1374/2127), validation 0,631 (159/252), integrità OK.
- `agentic/PLAYBOOK.md` (sezioni Definizioni, Candidati, P1), `lib/echonext_lib.py`.

## Cosa è stato fatto
1. Script `experiments/P1/baseline.py`. Smoke test con `debug_n=2000` (`python3 experiments/P1/baseline.py debug`, output in `results/P1/debug/`). Il dataset ha meno di 2000 righe per split in validation, quindi lo smoke test coincide quasi con il run completo. Poi run completo.
2. Due insiemi di feature (età+sesso; 7 feature ECG) × 5 modelli (`lib.model_zoo`), addestrati sul train e valutati su validation (AUROC con IC95% bootstrap, AUPRC).
3. Controllo di coerenza: CV a 5 fold raggruppata per `patient_key` sul train (`lib.oof_proba`), hist_gb con tutte le feature; in più, a scopo diagnostico, CV degli altri 4 modelli.
4. Permutation importance (AUROC, 20 ripetizioni) sul validation per il miglior modello con tutte le feature.
5. Salvato il candidato globale.

## Risultati
AUROC su **validation** (n=252, 159 positivi, prevalenza 0,631). Il test set non è stato toccato.

| Feature | Modello | AUROC | IC95% | AUPRC |
|---|---|---|---|---|
| età+sesso | mlp_dense (migliore) | 0,588 | 0,515–0,664 | 0,708 |
| età+sesso | hist_gb | 0,582 | 0,512–0,649 | 0,694 |
| età+sesso | logreg | 0,573 | 0,504–0,647 | 0,701 |
| età+sesso | lasso_logreg | 0,572 | 0,504–0,645 | 0,700 |
| età+sesso | random_forest | 0,571 | 0,516–0,640 | 0,711 |
| 7 feature ECG | lasso_logreg (migliore) | 0,574 | 0,497–0,649 | 0,700 |
| 7 feature ECG | mlp_dense | 0,572 | 0,502–0,654 | 0,709 |
| 7 feature ECG | logreg | 0,570 | 0,497–0,644 | 0,700 |
| 7 feature ECG | random_forest | 0,564 | 0,493–0,641 | 0,678 |
| 7 feature ECG | hist_gb | 0,492 | 0,424–0,568 | 0,644 |

Riferimento AUPRC: con prevalenza 0,631 un modello senza informazione ha AUPRC ≈ 0,63.

**TABULAR_SIGNAL = no.** Guadagno = 0,574 − 0,588 = **−0,013** (soglia ≥ 0,02) e limite inferiore IC95% con tutte le feature = **0,497** (soglia > 0,55). Entrambe le condizioni falliscono. Le AUROC sono tutte nell'intervallo 0,49–0,59, con IC che includono o sfiorano 0,5: sul validation le feature ECG tabulari non mostrano un segnale distinguibile dal caso e non aggiungono nulla a età+sesso.

**Controllo CV vs validation (hist_gb, tutte le feature):** AUROC CV (train, 2127 righe) = 0,600 (IC 0,579–0,624); validation = 0,492 (IC 0,424–0,568). Differenza 0,108 > 0,05: **discrepanza da segnalare**. Diagnostica sulla CV degli altri modelli (train): logreg 0,612, lasso 0,610, random_forest 0,613, mlp 0,593 (IC95% tipicamente ±0,025); su validation questi stessi modelli danno 0,564–0,574, cioè 0,02–0,04 in meno. Medie delle feature simili tra train e validation (es. età 60,0 vs 58,4; QRS 93,2 vs 93,8; sesso maschile 0,50 vs 0,46). Quindi: la differenza è più marcata per hist_gb, ma ho solo evidenze descrittive; non ho identificato la causa (con 252 pazienti l'IC del validation è ampio, ±0,07). Non ne deduco ragioni oltre i numeri. Anche la CV stessa è bassa (0,59–0,61), coerente con un segnale debole.

**Permutation importance** (validation, lasso_logreg, calo medio di AUROC ± sd):

| Feature | Calo medio | sd |
|---|---|---|
| age_at_ecg | 0,034 | 0,030 |
| atrial_rate | 0,021 | 0,009 |
| sex | 0,008 | 0,011 |
| pr_interval | 0,001 | 0,002 |
| ventricular_rate, qt_corrected | 0,000 | 0,000 |
| qrs_duration | −0,0004 | 0,028 |

Nessuna feature ha un'importanza chiaramente maggiore della sua variabilità, tranne forse atrial_rate (0,021 ± 0,009), valore piccolo in assoluto. Il modello lasso azzera di fatto 3 feature (coefficienti penalizzati L1).

## Decisioni e motivazioni
- Miglior modello per ciascun insieme scelto per AUROC su validation (come da PLAYBOOK): scelta fatta guardando il validation, quindi **ottimistica**; le differenze tra modelli (≤0,03) sono ampiamente entro gli IC.
- Candidato globale: modello migliore con tutte le feature ECG (lasso_logreg), addestrato sul train. Salvato nonostante il segnale assente, perché serve come riferimento in P7.
- Alternative: nessuna ottimizzazione di iperparametri (il segnale è debole e il validation andrebbe consumato inutilmente).

## Configurazioni provate
10 configurazioni (5 modelli × 2 insiemi di feature), tutte valutate su validation e registrate con `log-exp` (E0001, `--uses-val --n-configs 10`). La CV sul train e la permutation importance non scelgono nulla sul validation oltre al modello migliore già contato. Lo smoke test è stato eseguito sullo stesso validation (stessi 252 pazienti, stesse configurazioni) e non è contato a parte.

## Artefatti
- `experiments/P1/baseline.py`
- `results/P1/val_models.csv`, `results/P1/perm_importance.csv`, `results/P1/summary.json`, `results/P1/debug/`
- Candidato: `results/P1/candidates/global_best/` (`candidate.json`, `candidate.py`, `model.joblib`)

## Problemi aperti
- Discrepanza CV vs validation per hist_gb (0,600 vs 0,492), causa non identificata.
- Validation piccolo (252; IC larghi ±0,07): differenze <0,05 non sono interpretabili.
- Non è stata verificata l'origine dei dati (P0: cartella `/tmp/echonext_synth`, possibili dati sintetici; 380 righe `no_split`).
- Il diagnostico CV degli altri modelli non è salvato in un file (solo in questo report).

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- TABULAR_SIGNAL: no (guadagno −0,013 < 0,02; IC inferiore 0,497 < 0,55)
- AUROC_VAL_ETA_SESSO: 0,588 (IC95% 0,515–0,664), mlp_dense, n=252
- AUROC_VAL_TUTTE: 0,574 (IC95% 0,497–0,649), AUPRC 0,700, lasso_logreg
- MIGLIOR_MODELLO: lasso_logreg (7 feature ECG), scelto su validation (ottimistico)
- CV raggruppata hist_gb: 0,600 vs validation 0,492 (diff 0,108 > 0,05, causa non identificata)
- Importanza: age_at_ecg 0,034, atrial_rate 0,021; il resto ≈ 0
- CANDIDATO_GLOBALE: results/P1/candidates/global_best
- Configurazioni provate in P1: 10 (tutte su validation)
RACCOMANDAZIONE:
- Le feature tabulari ECG non hanno segnale sul composito: non aspettarsi sottogruppi forti dai soli tabulari; trattare ogni sottogruppo come esplorativo con correzione per multiplicità.
- Valutare se procedere con le forme d'onda (P4) come fonte principale di segnale.
