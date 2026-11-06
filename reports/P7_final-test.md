# Report P7 · final-test
- Agente: final-evaluator
- Data: 2026-11-06 15:55
- Stato: OK

## Obiettivo
Valutare UNA sola volta, sul test set (mai usato prima), i candidati congelati dopo la revisione P6, per stimare l'accuratezza reale fuori campione e verificare se la scelta fatta sul validation ha prodotto ottimismo. Ricordiamo che SHD (cardiopatia strutturale) non coincide con scompenso cardiaco.

## Input letti
- `python3 tools/state.py brief` (P6 done, PASS, approvazione umana P7 registrata, test_usato=False prima dell'esecuzione).
- `agentic/FROZEN.json`: cnn_multilabel (target SHD) e lvef_lte45_global (target lvef_lte_45_flag); comparatore `results/P1/candidates/global_best`. Coincidono con i `CANDIDATI_AMMESSI` di P6.
- Handoff P6 (pre-specificazione: riportare anche l'AUROC sulle righe con lvef_value noto; ignorare il comparatore per lvef_lte45_global).
- Definizione di "buona accuratezza" in `agentic/PLAYBOOK.md` riga 17: AUROC test ≥ 0.80 con limite inferiore IC95% ≥ 0.75; 0.70-0.80 moderata; <0.70 debole.

## Cosa è stato fatto
1. Verificato che P7 fosse approvata e i candidati coincidessero con quelli ammessi da P6.
2. Eseguito una sola volta `python3 tools/final_eval.py` (nessun errore tecnico, nessun rilancio). Nessun codice aggiuntivo, nessuna soglia modificata, nessun altro candidato provato. Nessuna riga di test è stata letta o stampata: solo aggregati.
3. Letti `results/final/test_results.json` e `reports/final_test_table.md`.
4. Nessuno smoke test (debug_n) applicabile: lo script è fornito dal driver e il test si apre una sola volta.

## Risultati
Valori calcolati sul TEST (n=5442 ECG, uno per paziente). IC95% tra parentesi quadre.

| Candidato | Target | AUROC test nello scope | AUROC modello globale P1 sugli stessi pazienti | AUROC test righe con LVEF noto | AUPRC test | AUROC val (P3/P4) | CALO_VAL_TEST | CLASSE_ACCURATEZZA |
|---|---|---|---|---|---|---|---|---|
| cnn_multilabel | SHD composito | 0.830 [0.820-0.841] (n=5442, pos=2318) | 0.762 [0.750-0.774] (hist_gb) | 0.817 [0.807-0.829] (n=4827, pos=2318) | 0.802 | 0.823 [0.811-0.834] | -0.007 (nessun calo; test +0.007) | buona |
| lvef_lte45_global | LVEF ≤ 45% | 0.793 [0.776-0.809] (n=5442, pos=962) | 0.775 [0.759-0.791] (non pertinente, vedi sotto) | 0.780 [0.763-0.797] (n=4827, pos=962) | 0.474 | 0.764 [0.748-0.782] | -0.029 (nessun calo; test +0.029) | moderata |

Note di lettura:
- Lo "scope" coincide con l'intero test (candidati globali, nessun sottogruppo/cluster): "test nello scope" = "test su tutti".
- CNN vs modello tabellare globale sugli stessi pazienti: +0.069 AUROC (0.830 vs 0.762), con IC95% marginali che non si sovrappongono (0.820-0.841 vs 0.750-0.774); differenza appaiata non calcolata dallo script, quindi l'IC della differenza non è riportato.
- Il comparatore P1 `global_best` è un modello SHD: per lvef_lte45_global il confronto (0.775) è dichiarato non pertinente da P6 e non va interpretato.
- Classe CNN: AUROC 0.830 ≥ 0.80 e limite inferiore 0.820 ≥ 0.75 → buona. Classe LVEF≤45: AUROC 0.793 < 0.80 → moderata (il limite inferiore 0.776 sarebbe sufficiente, ma la soglia puntuale no).
- Sulle sole righe con lvef_value noto (sottoinsieme pre-specificato da P6) l'AUROC cala di 0.013 (CNN: 0.830→0.817) e 0.013 (LVEF≤45: 0.793→0.780): le righe senza LVEF hanno tutti i flag a 0 per costruzione e gonfiano leggermente l'AUROC. Anche sul sottoinsieme la classe resta buona per la CNN (0.817, lo 0.807) e moderata per LVEF≤45 (0.780). La CNN è comunque più alta di 0.80 anche su questo sottoinsieme.
- Il validation aveva mostrato lo stesso calo sul sottoinsieme (CNN 0.812, LVEF≤45 0.749): la stima su test è coerente.
- Il CALO_VAL_TEST è negativo (il test è migliore): nessun segno di ottimismo da selezione (soglia 0.05). Un'eventuale differenza favorevole è compatibile col caso/differenza di casistica (prevalenza SHD 0.426 test vs 0.430 val); non va letta come miglioramento reale. Gli IC di val e test si sovrappongono per entrambi.

## Decisioni e motivazioni
- Nessuna ottimizzazione post-test: soglie, modelli e candidati invariati; risultati riportati così come sono.
- Come richiesto da P6 sono riportate sia l'AUROC su tutto il test sia quella sulle righe con LVEF noto; `final_eval.py` supporta quest'ultima (colonna `test_lvef_known`).
- Calo definito come AUROC val − AUROC test; valori negativi = nessun calo.

## Configurazioni provate
0 nuove configurazioni. 2 candidati valutati sul test, scelti prima (in P6) e congelati; nessuna selezione sul test. Nessun `log-exp` necessario (il test non è validation). Configurazioni su validation totali ereditate: 511 (brief).

## Artefatti
- `results/final/test_results.json`
- `reports/final_test_table.md`
- `agentic/FROZEN.json` (invariato)

## Problemi aperti
- Il test è ora usato (non riaprirlo): ogni ulteriore modifica ai candidati non sarebbe più valutabile in modo onesto.
- IC della differenza appaiata CNN vs globale non fornito da `final_eval.py`; il +0.069 è basato su due IC marginali.
- Un solo seed per la CNN; calibrazione non valutata; centro unico, solo pazienti con eco; i flag delle righe senza LVEF sono negativi per costruzione (AUROC su tutto il test leggermente gonfiata, vedi sottoinsieme).
- AUROC di lvef_lte45_global è moderata (0.793): non è un'accuratezza "buona" secondo la definizione di progetto.
- SHD non è scompenso cardiaco; l'ipotesi QRS resta inconclusiva (P3/P6) e non è stata testata in P7.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- cnn_multilabel (SHD) test: AUROC 0.830 [0.820-0.841], n=5442, pos=2318, AUPRC 0.802; globale P1 stessi pazienti 0.762 [0.750-0.774]
- cnn_multilabel: CALO_VAL_TEST -0.007 (val 0.823 → test 0.830); su righe con LVEF noto 0.817 [0.807-0.829]; CLASSE_ACCURATEZZA: buona
- lvef_lte45_global test: AUROC 0.793 [0.776-0.809], n=5442, pos=962, AUPRC 0.474; comparatore SHD non pertinente (0.775)
- lvef_lte45_global: CALO_VAL_TEST -0.029 (val 0.764 → test 0.793); righe con LVEF noto 0.780 [0.763-0.797]; CLASSE_ACCURATEZZA: moderata
- Nessun calo > 0.05: nessuna evidenza di ottimismo da selezione; 0 sottogruppi/cluster credibili da valutare
RACCOMANDAZIONE:
- P8: riportare CNN come buona (0.830, 0.817 sulle sole righe con LVEF), LVEF≤45 come moderata, con le cautele: SHD ≠ scompenso, un seed, centro unico, calibrazione non valutata, righe senza LVEF negative per costruzione.
- P8: dire chiaramente che nessun sottogruppo/cluster è risultato credibile e che QRS-modifica è inconclusiva; non riaprire il test.
