# PLAYBOOK della PIPELINE 2 (fasi Q0-Q10)

La pipeline 1 (P0-P9) ha trovato: segnale moderato nelle 7 misure ECG (AUROC 0.746), segnale buono nelle forme d'onda con una CNN (test 0.830), nessun sottogruppo/cluster credibile sulle sole misure tabulari, ipotesi QRS inconclusiva.
La pipeline 2 fa i test che restano possibili **con questi stessi dati** (nessun dato aggiuntivo è disponibile, nessuna informazione mancante è recuperabile) per arrivare a una conclusione.
Ogni fase scrive un report in `reports/Q<n>_<nome>.md` (formato di `agentic/REPORT_TEMPLATE.md`, convalida con `python3 tools/validate_report.py <file>`) con la sezione Handoff.

## Regole speciali della pipeline 2 (si sommano a CLAUDE.md)
1. **Comandi di stato**: usa `python3 tools/state2.py ...` (brief, handoff, note, log-exp, ...) e MAI `tools/state.py` (è la pipeline 1). Il registro `results/registry.csv` è condiviso: conta le configurazioni di entrambe le pipeline.
2. **Il test set è chiuso**. Era già stato usato una volta (P7). Lo aprirà solo `tools/final_eval2.py` in Q8, come *seconda lettura* pre-specificata, su pochi candidati scelti **senza** guardare il test. Nessun altro script lo carica; non usare `allow_test`.
3. **Pool di sviluppo = train + validation** (`pipe2_lib.load_dev`). Ogni scelta (iperparametri, derivazioni, soglie) si fa con la **cross-validation per paziente** (`fold` = hash di `patient_key`, 5 fold, identica in ogni script). Il validation non è più «a parte»: ogni configurazione provata si registra con
   `python3 tools/state2.py log-exp --phase Q<n> --desc "..." --uses-val --n-configs <N>` (`--uses-val` sempre, perché la CV usa anche il validation).
4. **Popolazione «clean»**: le righe senza `lvef_value` (circa 9-11%) hanno tutte le 12 etichette a 0 e PhysioNet dichiara che le eco senza LVEF sono state escluse dal dataset: sono anomale. Risultato principale = popolazione *clean* (LVEF noto); riporta sempre anche *all* come sensibilità. Non è possibile chiarire il motivo con altri dati: dichiaralo come limite.
5. **Lavori lunghi (CNN)**: i comandi di `tools/cnn_cv.py` sono riprendibili. Lancia SEMPRE in primo piano con `timeout` massimo del tool Bash (600000 ms) e `--time-budget-min 7`:
   `python3 tools/cnn_cv.py run --out results/Qn/nome ... --time-budget-min 7`. Se esce con codice 3 («STATO: PARZIALE») **rilancia lo stesso identico comando** finché stampa «STATO: COMPLETO».
   NON usare processi in background, `sleep`, `pgrep`, cicli di attesa: bloccano la fase. Dopo un'interruzione i job già fatti si saltano da soli.
6. **Un confronto = un numero con IC95% per paziente** (`pipe2_lib.auc_ci_cluster`, `auc_diff_ci_cluster`). Confronti multipli: correzione di Holm (`pipe2_lib.holm`) e dichiara quanti confronti hai fatto.
7. **Solo aggregati** nei report e nell'output: mai righe di paziente. Le predizioni per riga (OOF) restano in `results/`, non si stampano.
8. Definizioni (invariate): sottogruppo/cluster «credibile» = n ≥ 100, ≥ 25 positivi e ≥ 25 negativi, AUROC locale ≥ globale + 0.02 e **IC inferiore sopra l'AUROC globale**, riprodotto contro il rumore (permutazione, p < 0.05 dopo Holm). Classe di accuratezza (indicativa in CV, definitiva solo sul test): buona = AUROC ≥ 0.80 con IC inferiore ≥ 0.75; moderata 0.70-0.80; debole < 0.70.
9. Risultati negativi sono risultati: «non c'è segnale» va scritto con i numeri che lo mostrano (anche IC superiore), non forzato.
10. Idempotenza: se un file in `results/Q<n>/` esiste già e la configurazione coincide, non rifarlo. Tieni aggiornato `reports/Q<n>_<nome>.progress.md`.
11. Ordine delle derivazioni NON documentato: lo verifica Q0 (`LEAD_ORDER_OK`). Non scrivere nomi di derivazione (I, V1...) nei report se `LEAD_ORDER_OK` non è `yes`.

## Strumenti
- `lib/pipe2_lib.py`: `load_dev()`, `WaveReader`, `auc_ci_cluster`, `auc_diff_ci_cluster`, `boot_stat_cluster`, `reliability`, `ece`, `brier`, `holm`, `sens_spec_at`, `fold_of`.
- `tools/cnn_cv.py`: `run` (CV fold×seed, OOF + embedding + pesi di ogni job), `summarize` (AUROC con IC, confronto col tabulare, popolazioni clean/all), `final-fit`, `make-candidate`, e in Python `load_fold_model`, `make_net`, `predict_net`.
  Output di `run`: `oof_seed{s}_fold{f}.npz` (idx, preds[n,13], emb[n,256], mu, sd), `weights_seed{s}_fold{f}.pt`, `progress.json`, `config.json`. Colonne di `preds`: le 12 probabilità nell'ordine di `pipe2_lib.TARGETS` + LVEF stimata (colonna 12).
  Di `summarize`: `oof_seed{s}.npy`, `oof_ensemble.npy` (n_dev × 13, stesso ordine di `load_dev()`), `metrics_clean.csv`, `metrics_all.csv`, `summary.json`, `tabular_oof.npz` (hist_gb sulle 7 feature, stesse fold).
  **Embedding**: appartengono allo spazio della rete di *quel* job (seed, fold): non mescolare fold diversi.
- Gli script degli agenti vanno in `experiments/Q<n>/`; i risultati in `results/Q<n>/`.

---

### Q0 · inventory (inventory-auditor)
1. `python3 tools/state2.py brief`; leggi gli Handoff e i report della pipeline 1 (`reports/P*.md`, `reports/FINAL_REPORT.md`, `reports/final_test_table.md`). Verifica che esistano: `results/P4/` (pesi della CNN), `results/final/test_results.json`, candidati P1/P3/P4, `results/registry.csv`.
2. Salva `results/Q0/p1_facts.json` con i numeri chiave **copiati dai report** (non ricalcolati) e la loro fonte: AUROC tabulare e CNN (validation e test), prevalenze, N, configurazioni provate.
3. **Verifica dell'ordine delle derivazioni**: su ~2000 ECG di train (solo calcoli aggregati) regredisci ogni canale sugli altri: ipotesi standard (canali 0-11 = I, II, III, aVR, aVL, aVF, V1-V6): canale1 ≈ a·canale0 + b·canale2 con a, b > 0 (II = I + III); canale3 ≈ a·canale0 + b·canale1 con a, b < 0 (aVR); canale4 ≈ a·canale0 + b·canale2 con a > 0, b < 0 (aVL); canale5 ≈ a·canale1 + b·canale2 con a, b > 0 (aVF). Riporta R² e segni. `LEAD_ORDER_OK`: `yes` se tutte e 4 le relazioni hanno R² ≥ 0.95 e segni attesi; `partial` se solo alcune; `no` altrimenti. Le derivazioni toraciche non sono verificabili: dichiaralo.
4. Popolazione clean: N e frazione in train+val; fold: N pazienti per fold.
5. **Prova dei tempi**: `python3 tools/cnn_cv.py run --out results/Q0/bench --seeds 0 --epochs 2 --time-budget-min 0.01` (si ferma dopo il primo job; usa tutte le 5 fold del pool intero). Dai secondi per epoca stima: minuti per job con 12 epoche, per Q1 (5 fold × 3 seed), per Q2 (7 set di derivazioni × 5 fold) e per Q6 (final-fit 3 seed). GPU sì/no.
Handoff: `LEAD_ORDER_OK`, `R2_LEAD_RELATIONS`, `N_DEV`, `CLEAN_FRACTION`, `SEC_PER_EPOCH`, `EST_MIN_Q1`, `EST_MIN_Q2`, `EST_MIN_Q6`, `GPU`, `P1_ARTEFATTI_MANCANTI: [..]`.
Se manca un artefatto essenziale (CSV dei dati, waveforms), STATUS FALLITO con il motivo.

### Q1 · cnn-cv (cnn-cv-analyst)
Obiettivo: stima solida (con variabilità) delle prestazioni della CNN, finora basata su un solo addestramento.
1. Run principale: `python3 tools/cnn_cv.py run --out results/Q1/main --seeds 0,1,2 --epochs 12 --time-budget-min 7` (leads all, population clean) ripetuto fino a «STATO: COMPLETO». Poi `python3 tools/cnn_cv.py summarize --out results/Q1/main`.
   `log-exp` 1 configurazione (`--n-configs 1 --uses-val`).
2. Sensibilità alle righe senza LVEF: stessa configurazione con `--population all --seeds 0 --out results/Q1/all_rows`; riporta la differenza di AUROC (SHD, LVEF ≤ 45%) misurata sulle righe *clean*.
3. Esplorazione limitata (solo se `EST_MIN_Q1` di Q0 mostra che il tempo totale resta sotto circa 3 ore): al massimo 3 varianti con seed 0 (es. `--epochs 20`, `--lr 1e-3`, `--reg-weight 0`), ciascuna con `log-exp`. Una variante sostituisce la principale **solo se** migliora l'AUROC SHD *e* LVEF ≤ 45% di almeno 0.01 con IC appaiato della differenza che esclude 0; in quel caso rilancia con 3 seed in `results/Q1/best`. Altrimenti resta `main`. Non cercare ulteriormente.
4. Confronto con la pipeline 1: AUROC CV (clean) contro il validation di P4 (0.823) e contro il tabulare sulle stesse righe e fold.
5. Classi indicative per ognuna delle 12 etichette (buona/moderata/debole/non valutabile se < 100 positivi in clean).
Handoff: `CNN_DIR` (cartella da usare nelle fasi successive: main o best), `AUROC_SHD_CLEAN` (IC), `SEED_SD`, `TAB_AUROC_SHD_CLEAN`, `DIFF_SHD` (IC), `TARGETS_GOOD: [..]`, `TARGETS_MODERATE`, `TARGETS_WEAK`, `LVEF_REG: MAE/R2 CNN vs tabulare`, `SENSIBILITA_RIGHE_SENZA_LVEF: delta`, `CONFIG_FINALE: epochs, lr, reg_weight, leads`.

### Q2 · lead-ablation (lead-ablation-analyst)
Obiettivo: rimozione progressiva di derivazioni (l'idea originale «togliere input finché funziona»). Usa la configurazione `CONFIG_FINALE` di Q1; 1 seed (0), 5 fold; `population` come in Q1.
Insiemi: `all` (riusa il seed 0 di Q1: non rilanciare), `eight`, `limb`, `chest`, `I_II`, `II`, `I`, `I_II_V1`. Se `LEAD_ORDER_OK` ≠ `yes` usa solo gruppi di canali per indice (es. `0,1,2,3,4,5` e `6,7,8,9,10,11`, singoli canali) e dichiara che i nomi non sono verificati. Se il tempo stimato per Q2 supera circa 4 ore rinuncia prima a `I_II_V1` e `I`.
Per ogni insieme: `python3 tools/cnn_cv.py run --out results/Q2/<insieme> --seeds 0 --leads <insieme> ... --time-budget-min 7` fino a COMPLETO; `summarize`; `log-exp` (1 config per insieme).
Tabella `results/Q2/ablation_table.csv`: per SHD, LVEF ≤ 45%, e le 2 condizioni con AUROC più alto in Q1: AUROC, IC, differenza dall'insieme completo con IC appaiato.
Regola: «insieme sufficiente» = differenza puntuale ≥ −0.01 sia per SHD sia per LVEF ≤ 45% rispetto a tutte le derivazioni. Riporta il più piccolo.
Handoff: `LEAD_ORDER_OK`, `INSIEME_MINIMO_SUFFICIENTE`, `TABELLA` (righe sintetiche), `CALO_PER_INSIEME`, `NOTE_NOMI_DERIVAZIONI`.

### Q3 · lvef-qrs-cnn (lvef-qrs-cnn-analyst)
Obiettivo: ripetere l'ipotesi dei medici («il QRS spiega perché a parità di altre misure l'LVEF cambia») con un predittore molto migliore del tabulare. Usa `CNN_DIR/oof_ensemble.npy` (popolazione clean, ordine di `load_dev()`), `tabular_oof.npz`.
a) AUROC di LVEF ≤ 45% per fascia di QRS (<100, 100-119, ≥120): CNN e tabulare, differenza appaiata con IC, eterogeneità tra fasce.
b) Valore aggiunto del QRS sopra la CNN: regressione logistica `y ~ logit(cnn)` contro `y ~ logit(cnn) + qrs + qrs×logit(cnn)` con CV per paziente (stesse fold); ΔAUROC e Δlog-loss con IC per paziente.
c) LVEF continua: residuo = LVEF vera − stimata dalla CNN; associazione del residuo col QRS (Spearman, correlazione parziale dopo età, sesso, frequenza, PR, QTc) e ΔR² in CV aggiungendo QRS e interazione.
d) Coppie «discordanti» (un ECG per paziente): pazienti con LVEF stimata dalla CNN simile (differenza < 1 punto) ma LVEF vera diversa di ≥ 10 punti: differenza di QRS tra chi ha LVEF più bassa e chi più alta, con IC per paziente.
Holm sulle famiglie (a: 3 fasce; b: 2 metriche; c: 2; d: 1). `log-exp` (n-configs = numero di test).
Verdetto `QRS_MODIFICA`: `supportata` se (b) ΔAUROC con IC inferiore > 0 dopo Holm **e** (c) ΔR² ≥ 0.01 con IC inferiore > 0; `non_supportata` se (b) IC superiore < 0.01 e (c) IC superiore < 0.01; altrimenti `inconclusiva`. Scrivi i numeri che giustificano il verdetto.
Handoff: `QRS_MODIFICA`, `AUROC_PER_FASCIA`, `DELTA_AUROC_QRS` (IC), `DELTA_R2_QRS` (IC), `DIFF_QRS_COPPIE` (IC), `N_CONFRONTI`.

### Q4 · embedding-subgroups (embedding-analyst)
Obiettivo: ripetere la ricerca di cluster e sottogruppi con la rappresentazione appresa dalla CNN, e capire dove la CNN è migliore o peggiore.
1. **Cluster sugli embedding, fold per fold**: per `seed 0` e ciascun fold f, usa `emb` e `preds` di `oof_seed0_fold{f}.npz` (stessa rete, righe del fold, popolazione clean). Standardizza, PCA a 20 componenti, k-means con k ∈ {3, 4, 6, 8} (k scelto con la silhouette, senza etichette), stabilità tra due metà per paziente (ARI). Dentro ogni cluster: prevalenza di SHD, AUROC locale della *stessa* predizione contro quella globale del fold, differenza con IC appaiato per paziente. Rumore di fondo: 200 permutazioni delle etichette di cluster (dimensioni conservate), distribuzione del massimo `locale − globale`. Credibile solo con la definizione del PLAYBOOK §8.
   Replicazione: un tipo di cluster conta come «replicato» solo se un cluster con profilo clinico simile (età, sesso, QRS, frequenze) risulta credibile in almeno 2 fold. Dichiara il criterio usato.
2. **Strati pre-specificati** (CNN ensemble, popolazione clean): età (<50, 50-64, 65-74, ≥75), sesso, QRS (<100, 100-119, ≥120), frequenza ventricolare (terzili), PR e QTc (terzili) e ogni colonna demografica/di contesto presente nei metadati (elenca i **nomi** delle colonne con `list(dev.columns)`; escludi identificativi). AUROC per strato con IC, differenza appaiata dal resto dei pazienti; Holm su tutti gli strati. Per SHD e LVEF ≤ 45%.
3. **Regole sugli errori con cross-fitting**: albero di decisione (profondità ≤ 3, foglia minima 300) sull'errore assoluto della CNN, addestrato su una metà dei pazienti e valutato sull'altra (e viceversa); riporta le regole che reggono nella metà di verifica con margine e IC. Dichiara il numero di regole provate.
`log-exp` per ogni famiglia (n-configs = numero di cluster/strati/regole confrontati).
Handoff: `CLUSTER_CREDIBILI`, `CLUSTER_REPLICATI`, `P_EMPIRICO`, `STABILITA_ARI`, `STRATI_DOVE_LA_CNN_E_PEGGIO: [strato: AUROC IC, diff IC, p_Holm]`, `STRATI_DOVE_E_MIGLIORE`, `REGOLE_ERRORE_CHE_REGGONO`, `N_CONFRONTI`.

### Q5 · calibration-interpret (calibration-analyst)
a) Calibrazione delle probabilità OOF della CNN (clean) per SHD, LVEF ≤ 45% e le condizioni in `TARGETS_GOOD`: tabella di affidabilità a 10 bin, ECE e Brier con IC per paziente; confronto con ricalibrazione isotonica *fuori dal fold* (addestrata sui fold di sviluppo, applicata al fold lasciato fuori). Nota lo shift di prevalenza tra train (≈0.52) e validation (≈0.43): la calibrazione assoluta dipende dalla popolazione.
b) Punti operativi per SHD e LVEF ≤ 45%: soglie per sensibilità 90%, specificità 90% e massimo Youden, con sensibilità, specificità, PPV, NPV e IC; spiega che PPV/NPV valgono solo con quella prevalenza (pazienti con eco entro un anno, bias di selezione).
c) **Cosa guarda la rete** (solo aggregati): carica con `cnn_cv.load_fold_model` il modello del job (seed 0, fold 0) e usa fino a 5000 righe del suo fold (clean). Occlusione: azzera una derivazione alla volta (e poi una finestra temporale di 0.5 s alla volta, su tutte le derivazioni) e misura il calo di AUROC per SHD e LVEF ≤ 45% con IC. Ripeti sul fold 1 per verificare la coerenza. Nomi delle derivazioni solo se `LEAD_ORDER_OK = yes`.
`log-exp` per le analisi che usano più configurazioni.
Handoff: `ECE_SHD`, `ECE_LVEF`, `RICALIBRAZIONE_UTILE: si|no`, `PUNTI_OPERATIVI` (sintesi), `DERIVAZIONI_PIU_IMPORTANTI`, `FINESTRE_PIU_IMPORTANTI`, `COERENZA_FOLD0_FOLD1`.

### Q6 · candidates (candidate-freezer)
Obiettivo: scegliere con regole fissate PRIMA di guardare il test i candidati del secondo test, addestrarli e congelarli.
Regole di scelta (scrivile in `results/Q6/preregistration.md` prima di fare altro, con data e ora):
- C1: CNN con configurazione `CONFIG_FINALE` di Q1, uscita SHD (sempre). C2: stessa rete, uscita LVEF ≤ 45% (sempre).
- C3..: ogni altra etichetta in `TARGETS_GOOD` di Q1 con ≥ 100 positivi in clean, in ordine di AUROC; massimo `max_candidates_second_test` candidati in totale (config della pipeline 2).
- Se Q2 ha trovato un `INSIEME_MINIMO_SUFFICIENTE` più piccolo di tutte le derivazioni: un candidato in più con quel set, uscita SHD.
- Un candidato è «confermato» al test se AUROC test con IC inferiore ≥ 0.75 e calo rispetto all'AUROC CV (clean) ≤ 0.05; «buona» se AUROC ≥ 0.80 con IC inferiore ≥ 0.75. Scrivi questi criteri nel documento.
Procedura: `python3 tools/cnn_cv.py final-fit --out results/Q6/final_all --seeds 0,1,2 --epochs <..> --lr <..> --reg-weight <..> --leads all --time-budget-min 7` (ripeti fino a COMPLETO; analogo per l'insieme ridotto in `results/Q6/final_<insieme>`); poi per ogni candidato
`python3 tools/cnn_cv.py make-candidate --weights results/Q6/final_all --target <etichetta> --name <nome> --cand-out results/Q6/candidates/<nome> --cv-auroc <AUROC CV clean>`.
Verifica il contratto: `predict(df, lib)` su 40 righe di validation (solo forma e indice; il validation fa parte del training del final-fit).
Scrivi `agentic/FROZEN2.json`: `{"candidates": [{"name","dir","target","comparator_dir"}, ...], "rules": "<riassunto>"}`; `comparator_dir` = `results/P1/candidates/global_best` per SHD, `results/P3/candidates/lvef_lte45_global` per LVEF ≤ 45%, assente per gli altri. Poi `python3 tools/state2.py freeze` (registra le impronte). Dopo il congelamento NON toccare né i candidati né i pesi.
`log-exp`: 1 configurazione per final-fit.
Handoff: `CANDIDATI: [nome: target, set derivazioni]`, `REGOLE_PRESPECIFICATE: results/Q6/preregistration.md`, `CRITERI_CONFERMA`, `N_CANDIDATI`.

### Q7 · review2 (skeptic-reviewer2)
Revisione indipendente, scettica; **non ricalcoli con gli stessi script**: riprodurre con codice tuo. Controlli minimi:
1. Fold: nessun paziente è sia in addestramento sia nel fold di verifica (ricalcola dai dati con `fold_of`), per tutte le run `results/Q*/`.
2. Riproduzione: ricalcola con `sklearn.metrics.roc_auc_score` (non con le funzioni di pipe2_lib) l'AUROC da `oof_ensemble.npy` per almeno 4 etichette e il confronto con `metrics_clean.csv`: tolleranza 0.005.
3. Popolazioni: l'addestramento «clean» ha escluso le righe senza LVEF (`n_train` nei file `.npz`); le metriche «clean» le escludono.
4. Test chiuso: `results/final2/` non esiste ancora; nessuno script in `experiments/Q*/` contiene `allow_test`; `results/final/` viene solo da P7.
5. Multiplicità: conta le configurazioni nel registro (`results/registry.csv`) e i confronti dichiarati in Q3, Q4, Q5; verifica che Holm sia stata applicata dove dichiarato.
6. Affermazioni: per Q1-Q5 tabella `affermazione | supportata/debole/non supportata | motivo`.
7. Candidati: impronte coerenti (`python3 tools/state2.py freeze` è già stato eseguito; ricalcola sha256 senza modificare), contratto rispettato, regole di Q6 rispettate (numero, criteri scritti prima).
Esito `VERDETTO: PASS|REVISE`. Puoi solo **rimuovere** candidati da `agentic/FROZEN2.json` (mai aggiungerne o modificarne): se un candidato non regge, toglilo e spiegane il motivo. Se manca qualcosa di grave che non si risolve togliendo candidati: `STATUS: PARZIALE`, `VERDETTO: REVISE` e `REVISIONI_RICHIESTE: [fase: azione]`.
Handoff: `VERDETTO`, `CANDIDATI_AMMESSI`, `AFFERMAZIONI: n_supportate/n_deboli/n_non_supportate`, `DISCREPANZE_NUMERICHE`, `RISCHI_RESIDUI`.

### Q8 · final-test2 (final-evaluator2)
Richiede approvazione umana (`python3 tools/state2.py approve Q8`). Leggi `agentic/FROZEN2.json`, `reports/Q7_review2.md`. Esegui UNA volta `python3 tools/final_eval2.py` (se fallisce per un errore tecnico, non rilanciare alla cieca: riferisci l'errore). Se `CANDIDATI_AMMESSI` è vuoto, `python3 tools/state2.py skip Q8 --reason "nessun candidato"` e spiegalo.
Interpreta `results/final2/test_results.json` e `reports/final_test2_table.md`: per candidato AUROC test (IC), versione clean, differenza dal tabulare (IC), calo rispetto all'AUROC CV, ECE/Brier, classe e se è «confermato» secondo i criteri di Q6. Confronta con il primo test (P7: CNN SHD 0.830, LVEF ≤ 45% tabulare 0.793). Dichiara che è una **seconda lettura del test** e quanti candidati sono stati valutati. Non ottimizzare, non cambiare soglie, non riprovare.
Handoff: tabella del test, `CONFERMATI: [..]`, `NON_CONFERMATI: [..]`, `CALO_CV_TEST` per candidato, `CLASSE` per candidato.

### Q9 · synthesis2 (synthesizer2)
Report complessivo per i colleghi, **in parole povere**, su tutto il lavoro (pipeline 1 e 2). Scrivi `reports/Q9_synthesis2.md` (template standard, contenuto principale nella sezione «Risultati») e una copia autonoma `reports/REPORT_COMPLESSIVO.md` (senza le sezioni del template: titolo, indice, capitoli). Struttura:
1. **In breve** (≤ 12 righe): domanda, risposta, quanto fidarsi, c'è qualcosa di fattibile? (sì/no/in parte e per cosa).
2. **A. Qual era l'obiettivo degli esperimenti** (con il perché clinico e con le tre domande: SHD, singole condizioni, LVEF e QRS; ricorda che SHD ≠ scompenso cardiaco).
3. **B. Risultati ottenuti**: per obiettivo, con numeri e IC e classe di accuratezza; tabella riassuntiva *obiettivo → esito → fiducia*.
4. **C. Prove fatte, passo passo, in parole povere** (un paragrafo per passo, in ordine cronologico, prima pipeline 1: P0-P9 e poi pipeline 2: Q0-Q8). Ogni paragrafo risponde a: cosa volevamo capire, cosa abbiamo fatto (senza gergo), cosa abbiamo trovato (un numero), cosa abbiamo deciso di conseguenza. Esempio di stile: «Come primo passo abbiamo controllato i dati per capire se erano utilizzabili: …». Spiega ogni termine tecnico la prima volta.
5. **D. Cosa è fattibile e cosa no**, con i criteri usati per dirlo.
6. **E. Cosa si può ancora fare** SOLO con questi dati (nessun dato aggiuntivo disponibile) e cosa richiederebbe dati nuovi (per onestà, in una riga).
7. **F. Quanto fidarsi**: numero di configurazioni, selezione sul validation, revisione, due letture del test, limiti (un centro, solo pazienti con eco, righe senza LVEF, calibrazione, nessuna validazione esterna).
8. Glossario (8-12 termini) e **come riprodurre**.
Solo numeri presenti nei report o in `results/`; se due fonti discordano, riporta la discordanza.
Handoff: percorso del report, `VERDETTO_FATTIBILITA`, sintesi dei risultati chiave.

### Q10 · notebook2 (notebook-builder2)
Notebook complessivo che **integra quello della pipeline 1** (`reports/P9_notebook/replica_analisi.ipynb`) e aggiunge i capitoli della pipeline 2. Scrivi `reports/Q10_notebook/replica_completa.ipynb` (senza output), `LEGGIMI.md`, `requirements.txt`, e lo script `experiments/Q10/build_notebook2.py` che lo genera (parte dal JSON del notebook P9, ne riprende le celle e aggiunge quelle nuove; usa `nbformat` se c'è, altrimenti JSON v4.5).
Requisiti: parametri iniziali (`DATA_DIR`, `QUICK`, `SEED`, `RUN_HEAVY`); autonomo (non importa `lib/` né `tools/`: copia gli helper necessari, compresa la funzione di fold per paziente, il bootstrap per paziente e la definizione della CNN); test chiuso (il notebook non carica il test e non lo valuta; la tabella del secondo test è testo copiato da `reports/final_test2_table.md`); solo aggregati.
Capitoli nuovi, ciascuno con Obiettivo/perché, codice, risultato che abbiamo ottenuto (dai report, con fonte), decisione (con cella che ricalcola la regola dove possibile): Q0 protocollo e verifica delle derivazioni; Q1 CNN in CV; Q2 ablazione delle derivazioni; Q3 LVEF e QRS con la CNN; Q4 embedding, cluster e sottogruppi; Q5 calibrazione e interpretabilità; Q6-Q8 candidati, revisione e secondo test (testo). Le parti pesanti (CNN, occlusione, permutazioni) dietro `RUN_HEAVY` (False di default); con `QUICK=True` la CNN gira su un sottoinsieme piccolo con 1 seed e 2 epoche, solo per verificare che il codice funzioni.
Verifica: `ast.parse` su tutte le celle; esecuzione in-process (o `nbclient` se c'è) con `QUICK=True`; confronto di almeno 5 numeri con i report; tempi misurati. Se non riesci a eseguire qualcosa, STATUS PARZIALE e dillo.
Handoff: percorso, parametri, tempi, esito della verifica, cosa NON è replicabile nel notebook.
