# Report P9 · notebook
- Agente: notebook-builder
- Data: 2026-11-06 16:30
- Stato: PARZIALE

## Obiettivo
Trasformare ciò che la pipeline ha fatto (P0-P8) in un notebook Jupyter autonomo che i partner possano eseguire da soli, capitolo per capitolo, con obiettivo, scelte e «risultati attesi» presi dai report. Stato PARZIALE perché `nbformat`, `nbclient` e `ipykernel` non sono installati (vedi «Problemi aperti»).

## Input letti
- `python3 tools/state.py brief` e gli Handoff di P0-P8, R1-P3/P4/P5; `reports/P8_synthesis.md` (via handoff), `reports/final_test_table.md`, `reports/P9_notebook.progress.md`.
- Script: `experiments/P1/baseline.py`, `experiments/P3/lvef_qrs.py` (prime 80 righe), `experiments/P4/train.py` e `model_def.py`, `experiments/P5/select_cluster.py`; `lib/echonext_lib.py` (da cui sono copiati e semplificati gli helper); `results/P4/history.json`, `results/P1/val_models.csv`.
- Soglie: `agentic/pipeline_state.json` (config.thresholds). Dati: `/home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1` (solo train e validation; il test non è mai stato letto né contato).

## Cosa è stato fatto
1. `experiments/P9/build_notebook.py` genera `reports/P9_notebook/replica_analisi.ipynb` (40 celle, 15 di codice, nessun output salvato). Usa `nbformat` se c'è; poiché manca, costruisce lo stesso JSON nbformat v4.5 con un fallback (il notebook va comunque generato dallo script, non a mano).
2. Struttura: Preparazione (parametri `DATA_DIR` vuoto, `QUICK`, `SEED=42`, `RUN_HEAVY=False`; helper copiati da `lib/`: caricamento che scarta il test, AUROC con IC bootstrap, CV raggruppata, `show()` che rifiuta righe di paziente) → 1 Audit (P0) → 2 Baseline (P1) → 3 Condizioni (P2) → 4 LVEF e QRS (P3) → 5 Forme d'onda (P4, dietro `RUN_HEAVY`) → 6 Cluster (P5; permutazioni dietro `RUN_HEAVY`) → 7 Controlli del revisore (P6) → 8 Test finale (solo testo, tabella copiata da `final_test_table.md`), limiti e rinvio a `P8_synthesis.md`.
3. Ogni capitolo: Obiettivo/perché, codice, «Risultato che abbiamo ottenuto» (numeri e IC dai report, con il file di origine e l'avviso che con `QUICK=True` i numeri differiscono), «Decisione». Celle che ricalcolano le regole: segnale tabulare (guadagno ≥ 0.02, IC inf. > 0.55), scelta dei `cluster_targets`, assenza di P3b (ΔR² per fascia QRS), `waveform_dominant` (guadagno ≥ 0.05), cluster credibili.
4. Verifica: `experiments/P9/run_check.py` (a) `ast.parse` su tutte le celle di codice; (b) esecuzione sequenziale delle celle in un unico namespace Python (sostituto di `nbclient`, non disponibile) con `QUICK=True` e con `QUICK=False`, `RUN_HEAVY=False`; esito in `results/P9/run_check.json`. Il notebook consegnato è quello senza output.
5. Prova con `RUN_HEAVY=True`, `QUICK=False` (CNN 12 epoche, 30 permutazioni): `run_check.py full heavy`, 15/15 celle ok in 130.8 s su GPU RTX 5080; chiave `full_heavy` di `run_check.json`.

## Risultati
Esito della verifica (`results/P9/run_check.json`): ast.parse OK su 15/15 celle. Esecuzione `QUICK=True`: 15/15 celle ok in 10 s; esecuzione `QUICK=False` (RUN_HEAVY=False): 15/15 celle ok in 37 s.

Confronto con i report (validation; `QUICK=False` = esecuzione completa del notebook, `QUICK=True` = prova rapida):

| Numero | Report (fonte) | Notebook QUICK=False | Differenza | Notebook QUICK=True | Differenza |
|---|---|---|---|---|---|
| AUROC val hist_gb, 7 feature ECG (SHD) | 0.7464 (P1) | 0.7464 | 0.0000 | 0.7404 | −0.006 |
| AUROC val migliore età+sesso | 0.6587 (P1) | 0.6587 | 0.0000 | 0.6541 | −0.005 |
| AUROC CV raggruppata train, hist_gb | 0.7278 (P1) | 0.7278 | 0.0000 | 0.7226 | −0.005 |
| Guadagno sopra età+sesso | 0.0877 (P1) | 0.0877 | 0.0000 | 0.0864 | −0.001 |
| AUROC val stenosi aortica | 0.844 (P2) | 0.8435 | −0.0005 | 0.8347 | −0.009 |
| LVEF regressione MAE / R² | 9.54 / 0.163 (P3) | 9.54 / 0.163 | 0.00 / 0.000 | 9.71 / 0.131 | +0.17 / −0.032 |
| AUROC val LVEF ≤ 45% | 0.764 (P3) | 0.7643 | 0.0003 | 0.761 | −0.003 |
| Silhouette cluster (k=4) | 0.569 (P5) | 0.5692 | 0.0002 | 0.5625 | −0.007 |
| Stabilità ARI | 0.999 (P5, protocollo con 2 semi) | 0.9911 | −0.008 | 0.9985 | −0.001 |
| Cluster credibili | 0 (P5) | 0 | = | 0 | = |
| AUROC val hist_gb, sole righe con LVEF | 0.732 (P6) | 0.7318 | −0.0002 | 0.724 | −0.008 |
| Prevalenza SHD train / val | 0.524 / 0.430 (P0) | 0.5237 / 0.4302 | ≈0 | 0.527 / 0.430 | +0.003 / 0 |

Con `QUICK=False` i 12 numeri coincidono con i report entro 0.001 (tranne l'ARI, −0.008, dovuta alla procedura semplificata a una sola coppia di metà). Con `QUICK=True` le differenze sono ≤ 0.01 di AUROC e −0.03 di R²: tolleranza dichiarata ±0.01 per AUROC, ±0.04 per R². Tutti i numeri della tabella sono su validation tranne la CV (train).
Prova pesante: AUROC CNN riaddestrata 0.8276 contro 0.823 del report (P4): differenza +0.005, entro l'IC del report ([0.811-0.834]; un solo seed, DataLoader semplificato). Il solo output della cella permutazioni non è stato catturato dal runner (cella ok, 25 s), quindi il p empirico ricalcolato non è confrontato col 0.032 del report.
Tempi per il LEGGIMI: QUICK ≈ 10 s, completo senza parti pesanti ≈ 37 s, con RUN_HEAVY=True ≈ 131 s; CNN originale ≈ 4-6 s/epoca su GPU (`results/P4/history.json`).

## Decisioni e motivazioni
- **Notebook autonomo**: nessun import da `lib/` o `tools/`; helper copiati e semplificati (modelli: logreg + hist_gb in QUICK; + random forest e MLP in modalità completa; `lasso_logreg` omesso).
- **Test chiuso**: il caricamento scarta test e `no_split` subito dopo la lettura del CSV; il capitolo finale riporta solo la tabella di `final_test_table.md` come testo, con avviso «si valuta una sola volta».
- **Semplificazioni dichiarate nel notebook**: niente ricerca di 1 950 regole di sottogruppo (P2); del QRS solo il test (i) (ii e iii restano negli script); per i cluster feature e k fissati (sex, atrial_rate, qrs_duration; k=4) senza ripetere la selezione; permutazioni senza ripetere la selezione; revisione R1 su modelli salvati non riprodotta. Per ognuna il capitolo dice dove sta lo script.
- **QUICK** sottocampiona i *pazienti* di train (6 000), non le righe, per non rompere i gruppi; la validation resta intera.
- **Parti pesanti** (CNN, permutazioni) dietro `RUN_HEAVY=False` con i risultati attesi dai report; cap. 5 usa il valore del report nella cella di decisione se la CNN non è stata riaddestrata (dichiarato nell'output).
- Fasi saltate/fallite: nessuna (P5 era PARZIALE per le configurazioni non registrate, poi sanate in R1-P5; P3b e P5b non sono scattati: spiegato nelle celle «Decisione»).
- Nessuna numerica inventata: ogni «risultato atteso» riporta il report di origine.

## Configurazioni provate
Nessuna nuova configurazione sul validation per scopo di selezione: il notebook ripete modelli già registrati (E0001-E0022, 433 configurazioni con validation in totale, 511 complessive). Non è stato eseguito `log-exp` per P9: le esecuzioni di verifica (2, QUICK e completa; più la prova pesante) ricalcolano metriche già note e non scelgono nulla guardando il validation.

## Artefatti
- `reports/P9_notebook/replica_analisi.ipynb` (senza output), `reports/P9_notebook/LEGGIMI.md`, `reports/P9_notebook/requirements.txt` (numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, torch 2.15.0).
- `experiments/P9/build_notebook.py`, `experiments/P9/run_check.py`; `results/P9/run_check.json`.
- Segnalazione: `results/P9/run_quick.log` non esiste (redirect non consentito dall'ambiente).

## Problemi aperti
- **`nbformat`, `nbclient`, `ipykernel` non installati** (nessuna installazione, come da regole): il notebook è generato con un fallback JSON v4.5 e **non è stato eseguito con `nbclient`** né aperto in un kernel Jupyter. La verifica equivalente è l'esecuzione in-process delle celle (`run_check.py`). Il formato del file non è stato validato con `nbformat.validate`; ai partner conviene aprirlo e, se Jupyter protesta, rigenerarlo con `build_notebook.py` in un ambiente con `nbformat`.
- **Prova pesante**: eseguita senza errori (15/15 celle, 131 s, GPU). Non verificato: il comportamento su CPU (tempo non misurato) e il valore del p empirico delle permutazioni (output non catturato).
- Le AUROC con IC bootstrap usano 300 ricampionamenti (100 in QUICK): gli IC possono differire di ±0.002 dai report.
- Il LEGGIMI e il notebook dichiarano i tempi della sola macchina di verifica (GPU RTX 5080); su CPU non misurato.
- Dalla revisione P6 ereditati: SHD ≠ scompenso; righe senza LVEF negative per costruzione; calibrazione non valutata; un seed.

## Handoff
STATUS: PARZIALE
RISULTATI_CHIAVE:
- NOTEBOOK: reports/P9_notebook/replica_analisi.ipynb (40 celle, 15 di codice, senza output), con LEGGIMI.md e requirements.txt
- PARAMETRI: DATA_DIR (da compilare), QUICK=True, SEED=42, RUN_HEAVY=False
- TEMPI: QUICK ≈ 10 s; completo ≈ 37 s; RUN_HEAVY=True ≈ 131 s su GPU (CNN 0.8276 vs 0.823 del report); su CPU non misurato
- VERIFICA: ast.parse 15/15; esecuzione celle QUICK 15/15, completa 15/15, completa+heavy 15/15 ok (in-process, nbclient assente)
- CONFRONTO: 12 numeri; con QUICK=False differenza ≤ 0.001 (ARI −0.008); con QUICK=True ≤ 0.01 AUROC (R² −0.032)
- NON REPLICABILE NEL NOTEBOOK: test finale, 1 950 regole di sottogruppo, test QRS (ii)-(iii), selezione feature di clustering, revisione sui modelli salvati, CNN con RUN_HEAVY=False
RACCOMANDAZIONE:
- Installare nbformat/nbclient in un ambiente di verifica e rieseguire `run_check` (o nbclient) per chiudere lo stato PARZIALE; la prova `full_heavy` è già in run_check.json.
- Chiarire con i partner il significato delle righe senza LVEF prima di diffondere i numeri.
