# Replica completa dell'analisi EchoNext (pipeline 1 + pipeline 2): come usare il notebook

File: `replica_completa.ipynb` (consegnato **senza output salvati**; si rigenera con `python3 experiments/Q10/build_notebook2.py`, che legge il notebook P9 senza modificarlo e aggiunge i capitoli nuovi). Verifica: `python3 experiments/Q10/run_check2.py quick|full [heavy]` (esegue le celle in-process, `nbclient` non è installato).

## Cosa serve
1. **I dati EchoNext** (PhysioNet 1.1.1): cartella con il CSV dei metadati (nome contenente `metadata`) e `EchoNext_train_waveforms.npy` / `EchoNext_val_waveforms.npy` (questi servono per le CNN con `RUN_HEAVY=True` e per la verifica delle derivazioni, cap. 9). Il notebook non include né copia i dati.
2. **Python 3** con i pacchetti di `requirements.txt` (`torch` solo per `RUN_HEAVY=True`) e Jupyter (`ipykernel`, non importato dal codice).
3. Nessun altro file del progetto: il notebook **non importa `lib/` né `tools/`**. Gli helper (fold per paziente, bootstrap per paziente, Holm, ECE, rete CNN, lettore delle forme d'onda) sono copiati nelle celle dei cap. 0, 9 e 10.

## Come si esegue
1. Nella cella «Preparazione» scrivere `DATA_DIR` (o impostare `ECHONEXT_DIR`).
2. Prima `QUICK=True` (verifica che il codice giri), poi `QUICK=False`; `RUN_HEAVY=True` per CNN, permutazioni e occlusione.

## Parametri
| Parametro | Significato |
|---|---|
| `DATA_DIR` | cartella dei dati (da compilare) |
| `QUICK` | `True`: train a 6 000 pazienti (pipeline 1) e, per i cap. 9-14, 1 500 pazienti in tutto, 1 seed, 2 epoche, 50 ricampionamenti, 20 permutazioni, 300 righe per l'occlusione; **i numeri non sono quelli dei report** (solo verifica del codice). `False`: dati interi, 3 seed e 12 epoche per la CNN, 300 bootstrap, 200 permutazioni, 5000 righe per l'occlusione |
| `SEED` | 42 |
| `RUN_HEAVY` | `False` (default): nessuna CNN; i cap. 10-14 mostrano i numeri dei report e le regole di decisione ricalcolate su di essi. `True`: addestra le CNN in CV (cap. 10, 11), poi cap. 12-14 sulle predizioni ottenute |

## Tempi misurati (GPU RTX 5080; pool di sviluppo 77 101 ECG)
- `QUICK=True`, `RUN_HEAVY=False`: **12 s**.
- `QUICK=False`, `RUN_HEAVY=False`: **48 s** (di cui ~18 s la verifica delle derivazioni, ~10 s il tabulare in CV del cap. 10).
- `QUICK=True`, `RUN_HEAVY=True`: **51-57 s** (CNN 2 epoche su 1 500 pazienti; tutte le celle di codice eseguite senza errori).
- `QUICK=False`, `RUN_HEAVY=True`: **non misurato** (non eseguito). Stime dai report: Q1 ≈ 30 min (15 addestramenti da 12 epoche, ~9.5 s/epoca con preload in RAM; il lettore del notebook usa mmap senza preload e può essere più lento), Q2 ≈ 70 min, più Q3-Q5 e le permutazioni: nell'ordine di 2 ore. Su CPU: non misurato.

## Capitoli
0 Preparazione · 1 Audit (P0) · 2 Baseline (P1) · 3 Condizioni (P2) · 4 LVEF e QRS tabulare (P3) · 5 Forme d'onda (P4) · 6 Cluster (P5) · 7 Controlli del revisore (P6) · 8 Test finale pipeline 1 (testo) · **9 Protocollo e verifica delle derivazioni (Q0)** · **10 CNN in CV (Q1)** · **11 Ablazione delle derivazioni (Q2)** · **12 LVEF e QRS con la CNN (Q3)** · **13 Embedding, cluster, strati, regole sugli errori (Q4)** · **14 Calibrazione e interpretabilità (Q5)** · **15 Candidati, revisione, secondo test (Q6-Q8, testo)** · **16 Sintesi** · **17 Riproduzione, tempi, limiti**.
Ogni capitolo: Obiettivo/perché → codice → «Risultato che abbiamo ottenuto» (numeri dai report, con fonte) → «Decisione» (con celle che ricalcolano la regola: minimo insieme sufficiente, verdetto QRS, credibilità, conferma dei candidati).

## Numeri confrontati con i report (QUICK=False, RUN_HEAVY=False)
Pool di sviluppo 77 101 ECG (atteso 77 101) e 30 844 pazienti (30 844); frazione clean 0.9096 (0.910); R² aVF 0.9960 (0.996); AUROC tabulare SHD clean 0.7121 (0.712); calo CV→test C1 −0.009 (−0.009, ricalcolato dal testo copiato).

## Cosa NON è replicabile nel notebook
- **Il test set** (due letture: P7 e Q8): il notebook non lo carica; le tabelle sono testo copiato dai report.
- I numeri della CNN con `RUN_HEAVY=False` (presi dai report) e, con `QUICK=True`, qualsiasi numero della CNN.
- **La replica completa delle CNN non è stata rieseguita** in questa verifica: non è stato quindi confrontato il valore 0.815 della CNN né i risultati dei cap. 11-14 a pieno campione. Con più seed, mmap e senza i job di `tools/cnn_cv.py` (riprendibili, con preload) i numeri possono differire leggermente.
- La ricerca di 1 950 regole di sottogruppo (P5), il final-fit e la congelazione con sha256 (Q6), la revisione indipendente (Q7), le varianti di iperparametri di Q1, il registro delle 725 configurazioni, la sensibilità alle righe senza LVEF.

## Privacy
Solo aggregati: `show()` rifiuta tabelle con identificativi di paziente; nessun dato viene salvato su file.
