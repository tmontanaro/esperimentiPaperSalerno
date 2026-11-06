# Replica passo-passo dell'analisi EchoNext: come usare il notebook

File: `replica_analisi.ipynb` (consegnato **senza output salvati**; si rigenera con `python3 experiments/P9/build_notebook.py`).

## Cosa serve
1. **I dati EchoNext** (PhysioNet, versione 1.1.1): una cartella con il CSV dei metadati (nome che contiene `metadata`) e i file `EchoNext_train_waveforms.npy` / `EchoNext_val_waveforms.npy` (questi ultimi servono solo per la CNN con `RUN_HEAVY=True`). Il notebook non include né copia i dati.
2. **Python 3** con i pacchetti di `requirements.txt` (numpy, pandas, scikit-learn, scipy; `torch` solo per `RUN_HEAVY=True`) e un ambiente Jupyter (JupyterLab o Notebook con `ipykernel`; non sono importati dal codice).
3. Nessun altro file del progetto: il notebook **non importa `lib/` né `tools/`**; la prima cella di codice («Preparazione») contiene gli helper.

## Come si esegue
1. Aprire il notebook e, nella prima cella di codice, scrivere in `DATA_DIR` il percorso della cartella dei dati (nessun percorso personale è scritto nel codice; in alternativa impostare la variabile d'ambiente `ECHONEXT_DIR`).
2. Lasciare `QUICK=True` per una prova rapida, poi eseguire tutte le celle in ordine («Esegui tutto»).
3. Per l'analisi completa: `QUICK=False`. Per la CNN e le permutazioni dei cluster: `RUN_HEAVY=True`.

## Parametri (prima cella di codice)
| Parametro | Significato |
|---|---|
| `DATA_DIR` | Cartella dei dati (da compilare). |
| `QUICK` | `True`: train ridotto a 6 000 pazienti (la validation resta intera), solo logreg + hist_gb, 100 ricampionamenti bootstrap, CV a 3 fold, CNN con 3 epoche. `False`: train intero, 4 modelli, 300 bootstrap, CV a 5 fold, CNN a 12 epoche. |
| `SEED` | 42, fisso come nella pipeline. |
| `RUN_HEAVY` | `False` (default): CNN e permutazioni dei cluster **non** vengono eseguite; il notebook mostra i risultati attesi dai report. `True`: le esegue (serve `torch`, meglio con GPU). |

## Tempi (misurati su macchina con GPU RTX 5080, 72 475 ECG di train)
- `QUICK=True`, `RUN_HEAVY=False`: **circa 10 secondi** (misurato).
- `QUICK=False`, `RUN_HEAVY=False`: **circa 37 secondi** (misurato).
- `QUICK=False`, `RUN_HEAVY=True`: **circa 131 secondi** con GPU (CNN a 12 epoche ≈ 68 s, permutazioni dei cluster ≈ 25 s; misurato). AUROC della CNN riaddestrata nella prova: 0.8276 (report: 0.823 [0.811-0.834], un solo seed). Su CPU: non misurato.

## Che cosa contiene (un capitolo per fase)
0 Preparazione · 1 Audit dati (P0) · 2 Baseline (P1) · 3 Condizioni (P2) · 4 LVEF e QRS (P3) · 5 Forme d'onda (P4, pesante) · 6 Cluster (P5) · 7 Controlli del revisore (P6) · 8 Test finale (testo, P7), sintesi e limiti (P8).
Ogni capitolo: Obiettivo e perché → codice → «Risultato che abbiamo ottenuto» (numeri dai report) → «Decisione» (con una cella che ricalcola la regola, es. guadagno sopra età+sesso ≥ 0.02).

## Cosa NON è replicabile nel notebook
- La **valutazione sul test set**: si fa una sola volta (fase P7) e non va riaperta; il notebook riporta solo la tabella finale come testo.
- La ricerca di **1 950 regole di sottogruppo** (P2), i test (ii) interazione e (iii) coppie di pazienti simili sul QRS (P3), la **selezione delle feature di clustering** (P5) e la revisione sui modelli salvati (P6): sono negli script `experiments/P2`, `P3`, `P5`, `R1-*`.
- I numeri della CNN con `RUN_HEAVY=False` (sono presi dai report).
- La CNN riaddestrata può differire dal report (un solo seed; la pipeline originale usava DataLoader con più processi e un ordine dei lotti diverso).

## Privacy
Il notebook stampa solo aggregati (conteggi, metriche, tabelle di sintesi); non salva dati su file e `show()` rifiuta tabelle con identificativi di paziente.
