# Progetto EchoNext: ricerca di sottogruppi in cui la predizione funziona

## Scopo
Dal dataset EchoNext (12-lead ECG + etichette di cardiopatia strutturale, SHD) vogliamo individuare **gruppi di feature e/o di pazienti**
per cui gli obiettivi sotto sono predicibili con buona accuratezza, e dire con onestà dove NON lo sono.
Obiettivi: (1) `shd_moderate_or_greater_flag` (composito); (2) le singole condizioni (`*_flag`); (3) `lvef_value` / `lvef_lte_45_flag`,
con l'ipotesi dei medici che `qrs_duration` modifichi la relazione tra feature e LVEF.
Attenzione: SHD NON coincide con scompenso cardiaco. Ricordarlo nei report.

## Regole non negoziabili
1. **Lo stato sta su disco, non nella tua memoria.** All'inizio di ogni sessione esegui `python3 tools/state.py brief`.
   Il contesto può esaurirsi in qualsiasi momento: tutto ciò che serve per ripartire deve essere già in `reports/`, `results/`, `agentic/`.
2. **Il test set è chiuso.** Lo apre solo `tools/final_eval.py` (fase P7). Non leggere, contare né descrivere righe di test. `lib/echonext_lib.py` lo impedisce, non aggirarlo.
3. **Niente leakage.** Input dei modelli: solo `ECG_FEATURES` e forme d'onda. Mai colonne derivate dall'eco (`ECHO_ONLY_COLS`), mai l'etichetta o suoi derivati.
   Cross-validation sempre raggruppata per `patient_key` (`GroupKFold`).
4. **Il validation set non si "consuma".** Ogni configurazione provata che usa il validation va registrata:
   `python3 tools/state.py log-exp --phase <ID> --desc "..." --uses-val --n-configs <N>`. Un risultato scelto guardando il validation è ottimistico: dillo.
5. **Privacy: solo aggregati.** Non stampare né leggere righe a livello di paziente (niente `head()` del dataset, niente `cat` dei CSV). Usa `lib.show(df)`.
   Non aprire con lo strumento Read i file `.csv`/`.npy` del dataset. I dati escono dalla macchina solo come numeri aggregati nei report.
6. **Ogni affermazione ha un numero e un intervallo di confidenza.** AUROC sempre con IC95% (`lib.metrics_binary`). Un sottogruppo è "credibile" solo secondo le definizioni di `agentic/PLAYBOOK.md`.
7. **Risultati negativi sono risultati.** "Non c'è segnale" è un esito valido e utile ai partner. Non forzare l'analisi finché qualcosa "funziona".
8. **Passi idempotenti.** Salva i risultati intermedi in `results/<ID>/` e salta ciò che esiste già. Tieni `reports/<ID>_<nome>.progress.md` aggiornato a ogni passo importante (cosa fatto, cosa resta): è ciò che permette di riprendere dopo un'interruzione.
9. **Prima in piccolo.** Ogni script gira prima con `debug_n=2000` (smoke test), poi sull'intero dataset.
10. **Non modificare** `lib/`, `tools/`, `.claude/`, `agentic/PLAYBOOK.md`, `agentic/pipeline_state.json` (usa `tools/state.py`).

## Protocollo di ogni fase
1. `python3 tools/state.py brief`, poi `python3 tools/state.py handoff <ID>` per ogni fase da cui dipendi (leggi solo gli Handoff, non i report interi, salvo necessità).
2. Leggi la tua sezione in `agentic/PLAYBOOK.md`. I notebook in `legacy_notebooks/` sono ricette di partenza: portale in script sotto `experiments/<ID>/`.
3. Lavora a passi piccoli; aggiorna il file `.progress.md`; salva output in `results/<ID>/`; registra le configurazioni con `log-exp`; annota le decisioni con
   `python3 tools/state.py note --kind decision --phase <ID> --text "..."`.
4. Scrivi il report `reports/<ID>_<nome>.md` seguendo `agentic/REPORT_TEMPLATE.md` e convalidalo: `python3 tools/validate_report.py <file>`.
5. Non segnare tu la fase come completata: lo fa il driver (`tools/run_pipeline.py`) o l'orchestratore.
6. Salva i candidati finali (modello/sottogruppo/cluster) con `lib.save_candidate(...)`: contratto in `agentic/PLAYBOOK.md` §Candidati.

## Mappa dei file
- `agentic/PLAYBOOK.md` fasi, soglie, regole di decisione (gate) · `agentic/REPORT_TEMPLATE.md` formato dei report
- `agentic/pipeline_state.json` stato (solo via `tools/state.py`) · `agentic/DECISIONS.md` decisioni · `agentic/RUNLOG.md` log del driver · `agentic/FROZEN.json` candidati congelati
- `lib/echonext_lib.py` caricamento dati sicuro, metriche, modelli · `tools/` state, driver, validatore, valutazione finale
- `reports/` un report per fase · `results/` output, `results/registry.csv` esperimenti provati · `experiments/` script degli agenti

## Convenzioni
- Lingua dei report: italiano, chiaro per chi non è esperto di ML (spiega i termini la prima volta).
- Seed fisso 42. Python: `python3`. Librerie già installate; non installarne altre (se manca qualcosa scrivilo in "Problemi aperti").
- Metrica principale: AUROC (con AUPRC); per LVEF continuo MAE e R².
