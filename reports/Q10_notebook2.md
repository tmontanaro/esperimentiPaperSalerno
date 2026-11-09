# Report Q10 · notebook2
- Agente: notebook-builder2
- Data: 2026-11-09 18:00
- Stato: OK

## Obiettivo
Costruire un unico notebook autonomo con cui i colleghi rifanno passo passo tutte le analisi (pipeline 1 e 2): ogni capitolo ha obiettivo e perché, codice, risultato ottenuto (con la fonte nei report) e decisione. Il notebook è utile anche come documento: dice dove la CNN funziona (SHD, LVEF ≤ 45%, VD, stenosi aortica) e dove no (sottogruppi, QRS, LVEF numerica precisa). SHD non coincide con scompenso cardiaco.

## Input letti
- `python3 tools/state2.py brief` e handoff Q9 (verdetto «IN PARTE»; numeri del test clean con IC95% per paziente; 725 configurazioni; test letto due volte).
- `agentic/PLAYBOOK2.md` (sezione Q10 e regole speciali), `reports/P9_notebook/replica_analisi.ipynb` (letto, **non modificato**: 40 celle), `experiments/P9/run_check.py` (schema della verifica).
- Script delle fasi Q1-Q5 (`tools/cnn_cv.py`, `lib/pipe2_lib.py`, `experiments/Q3/qrs_cnn.py`, `Q4/emb.py`, `Q5/calib_ops.py`, `Q5/occlusion.py`) da cui sono copiati gli helper e il codice; `results/Q1/main/metrics_clean.csv`, `results/Q2/ablation_table.csv` e gli handoff Q0-Q9 per i numeri attesi; `reports/final_test2_table.md` (testo della tabella del secondo test).

## Cosa è stato fatto
1. Script `experiments/Q10/build_notebook2.py`: legge il JSON del notebook P9, ne riprende le celle 1-38 (cap. 0-8, con piccole modifiche di titolo e di descrizione dei parametri) e aggiunge 29 celle nuove (cap. 9-17). Nessun `nbformat` disponibile: scrive JSON v4.5 senza output. Rieseguibile.
2. Capitoli nuovi, ciascuno con Obiettivo/perché, codice, risultato (fonte: report e handoff) e decisione: 9 Q0 protocollo e verifica delle derivazioni; 10 Q1 CNN in CV e confronto con il tabulare; 11 Q2 ablazione (regola «insieme sufficiente» ricalcolata); 12 Q3 LVEF e QRS (a-d, verdetto ricalcolato); 13 Q4 cluster sugli embedding, strati, regole sugli errori (regola di credibilità); 14 Q5 calibrazione, punti operativi, occlusione; 15 Q6-Q8 candidati, revisione e secondo test (testo, criteri di conferma ricalcolati sui numeri copiati); 16 sintesi e limiti; 17 riproduzione, tempi, cosa non è replicabile.
3. Parametri iniziali nella cella 0: `DATA_DIR`, `QUICK`, `SEED`, `RUN_HEAVY` (False di default). Con `QUICK=True` e `RUN_HEAVY=True` la CNN gira su 1 500 pazienti, 1 seed, 2 epoche.
4. Autonomia: nessuna importazione di `lib/` o `tools/`; sono copiati fold per paziente (`fold_of`), bootstrap per paziente, Holm, ECE/Brier, la definizione della CNN, il lettore delle forme d'onda, la CV.
5. Test chiuso: il notebook parte da `load_data` di P9, che scarta le righe di test subito dopo la lettura; nessuna cella lo valuta. La tabella del secondo test è testo copiato da `reports/final_test2_table.md`.
6. Verifica (`experiments/Q10/run_check2.py`, esecuzione in-process, `nbclient` non installato): `ast.parse` su 26 celle di codice, notebook senza output salvati, esecuzione in 4 modalità (vedi Risultati). `LEGGIMI.md` e `requirements.txt` scritti a mano (non generati dallo script).
Smoke test: la modalità `QUICK=True` è lo smoke test; l'intero dataset è stato usato in `QUICK=False, RUN_HEAVY=False`.

## Risultati
Tempi misurati (GPU RTX 5080, pool di sviluppo 77 101 ECG):

| Modalità | Tempo | Esito |
|---|---|---|
| QUICK=True, RUN_HEAVY=False | 12 s | 26/26 celle ok |
| QUICK=False, RUN_HEAVY=False | 48 s | 26/26 celle ok |
| QUICK=True, RUN_HEAVY=True | 51-57 s | 26/26 celle ok |
| QUICK=False, RUN_HEAVY=True | non eseguito | stima dai report: nell'ordine di 2 ore |

Confronto con i report (QUICK=False, pool di sviluppo, nessun IC perché sono conteggi o valori puntuali di verifica; l'AUROC del tabulare è su CV per paziente, clean):

| Numero | Report | Ricalcolato nel notebook |
|---|---|---|
| ECG nel pool di sviluppo (Q0) | 77 101 | 77 101 |
| Pazienti nel pool di sviluppo (Q0) | 30 844 | 30 844 |
| Frazione clean (Q0) | 0.910 | 0.9096 |
| R² della relazione aVF ≈ f(II, III) (Q0) | 0.996 | 0.9960 |
| AUROC tabulare SHD, clean (Q1) | 0.712 [0.705-0.721] | 0.7121 |
| Calo CV→test di C1 (Q8, da testo copiato) | −0.009 | −0.009 |

- Le regole di decisione ricalcolate sui numeri dei report danno gli stessi esiti dichiarati: insieme minimo sufficiente = I_II_V1 (Q2); verdetto QRS = `non_supportata` con ΔAUROC IC sup +0.0006 e ΔR² IC sup +0.0010 (Q3); 5 candidati su 5 confermati e di classe «buona» (Q8).
- In modalità QUICK con CNN i numeri sono molto più bassi dei report (per esempio SHD clean 0.682 contro 0.815; 0 cluster credibili come atteso): è voluto (1 500 pazienti, 2 epoche) e serve solo a provare che il codice gira, **non** a replicare i valori.
- **Non verificato**: l'AUROC della CNN a pieno campione (0.815), i risultati completi dei cap. 11-14 (ablazione, QRS, cluster, calibrazione, occlusione) a pieno campione e il tempo di `QUICK=False, RUN_HEAVY=True`.

## Decisioni e motivazioni
- Il notebook P9 è riusato leggendolo (PLAYBOOK2 vieta di modificarlo): le sue celle 1-38 sono copiate; la cella finale del tempo è sostituita da una cella di confronto numeri e tempo.
- Le parti pesanti (CNN, occlusione, permutazioni) stanno dietro `RUN_HEAVY`; i capitoli 10-14 mostrano comunque i numeri dei report e ricalcolano le regole su di essi.
- Non è stata fatta la replica completa delle CNN: richiederebbe ore e il PLAYBOOK2 chiede solo la prova con `QUICK=True`. Questo limite è dichiarato nel notebook (cap. 17) e in `LEGGIMI.md`.
- Nessun nome di derivazione toracica nei testi: si usano nomi come «I_II_V1» solo come etichette dei set dei report; si precisa che il nome del canale 6 non è verificato (`LEAD_ORDER_OK = yes` vale per i canali 0-5).
- Il lettore delle forme d'onda del notebook usa mmap senza preload: più semplice, ma può essere più lento dei job di `tools/cnn_cv.py`.

## Configurazioni provate
Nessuna nuova configurazione di analisi: il notebook ricalcola analisi già registrate e le prove QUICK servono a verificare il codice, non a scegliere modelli. Il validation non è stato usato per scegliere nulla in questa fase, quindi nessun `log-exp`. Il totale nel registro resta quello del brief (725 configurazioni, 40 esperimenti).

## Artefatti
- `reports/Q10_notebook/replica_completa.ipynb` (68 celle, 26 di codice, senza output), `reports/Q10_notebook/LEGGIMI.md`, `reports/Q10_notebook/requirements.txt`
- `experiments/Q10/build_notebook2.py` (generatore), `experiments/Q10/run_check2.py` (verifica)
- `results/Q10/run_check.json` (esito celle, tempi, confronto numeri)
- `reports/Q10_notebook2.progress.md`

## Problemi aperti
- La replica completa con CNN (`QUICK=False, RUN_HEAVY=True`) non è stata eseguita: serve un'esecuzione di circa 2 ore su GPU per confrontare 0.815 (CNN), la classe delle 12 etichette e i capitoli 11-14 con i report. Fino ad allora quei capitoli sono verificati solo come codice funzionante.
- Con `RUN_HEAVY=True` il cap. 5 (P9) e il cap. 10 addestrano modelli diversi: i numeri possono differire leggermente dai report (seed, ordine dei lotti, mmap).
- Il test (due letture: P7 e Q8) non è replicabile nel notebook, per costruzione.
- `LEGGIMI.md` e `requirements.txt` non sono generati dallo script: se si rigenera il notebook con un diverso numero di celle va aggiornato il LEGGIMI a mano.
- `nbformat`/`nbclient` non installati: la verifica è un'esecuzione in-process delle celle, non in un kernel Jupyter.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- Notebook: `reports/Q10_notebook/replica_completa.ipynb` (68 celle, 26 di codice; cap. 0-8 da P9, cap. 9-17 nuovi), con `LEGGIMI.md` e `requirements.txt`; generato da `experiments/Q10/build_notebook2.py`.
- Parametri: `DATA_DIR`, `QUICK` (True di default), `SEED` 42, `RUN_HEAVY` (False di default).
- Tempi: QUICK 12 s; completo senza CNN 48 s; QUICK con CNN 51-57 s; completo con CNN non eseguito (stima circa 2 ore).
- Verifica: `ast.parse` ok, 26/26 celle in 3 modalità; 6 numeri ricalcolati uguali ai report (77 101 ECG, 30 844 pazienti, frazione clean 0.9096 contro 0.910, R² 0.9960 contro 0.996, tabulare SHD 0.7121 contro 0.712, calo C1 −0.009).
- NON replicabile nel notebook: test (P7, Q8: solo testo), CNN a pieno campione (non rieseguita), 1 950 regole P5, final-fit e sha256 Q6, revisione Q7, registro configurazioni.
RACCOMANDAZIONE:
- Per l'uso: partire con `QUICK=True`, poi `QUICK=False`; `RUN_HEAVY=True` solo con GPU e qualche ora a disposizione.
- Se serve la conferma numerica delle CNN, lanciare una volta `python3 experiments/Q10/run_check2.py full heavy` e confrontare i numeri con i report prima di diffondere il notebook come «replica verificata».
