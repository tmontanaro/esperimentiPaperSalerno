# Report P2 · conditions
- Agente: condition-analyst
- Data: 2026-11-06 14:50
- Stato: OK

## Obiettivo
Capire per quali condizioni cardiache (11 singole + composito) l'ECG tabellare (7 feature) permette di predire, se la predizione cambia tra gruppi noti (QRS, sesso, età, setting, etnia) e se esistono sottogruppi definiti da regole semplici in cui funziona meglio. Promemoria: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- Handoff P1: AUROC_val composito (hist_gb, 7 feature) 0.7464 [0.7331-0.7595]; TABULAR_SIGNAL yes; shift di prevalenza train 0.524 vs val 0.430 (gate P0).
- `agentic/PLAYBOOK.md` (sezione P2, definizione di "credibile"), soglie: min_n_val 100, min_pos_val 25, margine 0.02, soglia target promettente IC inferiore 0.65.
- Codice di riferimento: `legacy_notebooks/01_SHD_per_condizione_e_sottogruppi.ipynb` (ricetta); implementazione in `experiments/P2/conditions.py`.

## Cosa è stato fatto
Tutto in `experiments/P2/conditions.py` (modelli addestrati solo su train; input solo le 7 `ECG_FEATURES`; `location_setting` e `race_ethnicity` usate solo per stratificare).
- Smoke test `debug_n=2000` (usa il validation, registrato: 148 valutazioni), poi run completo.
- (a) 12 target × 3 modelli (logreg, hist_gb, mlp_dense), AUROC su validation con IC95% bootstrap (`lib.metrics_binary`). Nessun target saltato (soglia: ≥30 positivi in train e ≥10 in validation; il minimo è 21 positivi val per pulmonary_regurgitation).
- (b) per composito e per i 2 target promettenti di rango più alto (aortic stenosis, mitral regurgitation): modello globale hist_gb vs modello locale hist_gb (addestrato solo sul train di quello strato), sugli stessi pazienti di validation, con IC95% bootstrap appaiato sulla differenza locale−globale. Strati: qrs_bin, sesso, age_bin, location_setting, race_ethnicity (esclusi quelli con n<100, positivi<25 o negativi<25 in val).
- (c) 650 regole per target (38 atomi: 6 feature continue × 3 quantili del train (25/50/75%) × 2 direzioni + sesso; singole + coppie AND su feature diverse), per 3 target (composito, aortic stenosis, mitral regurgitation) = 1950 regole provate sul train con predizioni out-of-fold (hist_gb, GroupKFold 5 per `patient_key`). Regole eleggibili: n≥500, positivi≥50, negativi≥50. Ordinate per guadagno di AUROC OOF vs globale; le prime 5 per target (15 totali) verificate su validation.

## Risultati
**(a) Validation, AUROC [IC95%] del miglior modello per target** (3 modelli in `results/P2/a_conditions.csv`; la scelta del miglior modello usa il validation, quindi è leggermente ottimistica):

| Target | n_pos val | miglior modello | AUROC [IC95%] | AUPRC | promettente |
|---|---|---|---|---|---|
| aortic_stenosis | 252 | hist_gb | 0.844 [0.819-0.868] | 0.265 | sì |
| mitral_regurgitation | 282 | logreg | 0.770 [0.741-0.793] | 0.187 | sì |
| lvef_lte_45 | 866 | hist_gb | 0.764 [0.748-0.782] | 0.443 | sì |
| rv_systolic_dysfunction | 368 | hist_gb | 0.764 [0.737-0.787] | 0.242 | sì |
| tricuspid_regurgitation | 305 | mlp_dense | 0.756 [0.730-0.783] | 0.224 | sì |
| shd_moderate_or_greater (composito) | 1990 | hist_gb | 0.746 [0.733-0.759] | 0.699 | sì |
| pasp_gte_45 | 581 | mlp_dense | 0.688 [0.665-0.709] | 0.226 | sì |
| tr_max_gte_32 | 267 | logreg | 0.688 [0.656-0.720] | 0.102 | sì (IC inf 0.656) |
| lvwt_gte_13 | 877 | hist_gb | 0.686 [0.669-0.705] | 0.326 | sì (IC inf 0.669) |
| pericardial_effusion | 52 | logreg | 0.678 [0.604-0.750] | 0.023 | no (IC inf 0.604) |
| aortic_regurgitation | 62 | logreg | 0.674 [0.592-0.736] | 0.032 | no |
| pulmonary_regurgitation | 21 | hist_gb | 0.613 [0.500-0.751] | 0.010 | no (21 pos <25) |

Solo la stenosi aortica supera 0.80 di AUROC (con IC inferiore 0.819 sopra 0.75). Le altre condizioni sono "moderate" (0.70-0.80) o "deboli" (<0.70); la soglia di "buona accuratezza" si giudica solo sul test (P7). AUPRC bassi per le condizioni rare (prevalenza val: stenosi aortica 5.4%, MR 6.1%): AUROC alto non implica un buon valore predittivo. Le tre regurgitazioni/versamento con pochi positivi hanno IC larghi.

**(b) Composito, validation: globale vs locale sugli stessi pazienti** (`results/P2/b_strata.csv`; differenza = locale − globale):

| Strato | n val | pos val | AUROC globale [IC] | AUROC locale | diff [IC95%] |
|---|---|---|---|---|---|
| QRS <100 | 3388 | 1204 | 0.704 [0.689-0.722] | 0.704 | 0.000 [-0.003, 0.003] |
| QRS 100-119 | 671 | 379 | 0.723 [0.683-0.759] | 0.715 | -0.008 [-0.023, 0.005] |
| QRS >=120 | 567 | 407 | 0.752 [0.710-0.798] | 0.744 | -0.008 [-0.024, 0.009] |
| sesso F | 2356 | 869 | 0.739 [0.719-0.759] | 0.739 | 0.000 [-0.005, 0.005] |
| sesso M | 2270 | 1121 | 0.736 [0.718-0.754] | 0.729 | -0.007 [-0.012, -0.002] |
| età <50 | 969 | 264 | 0.717 [0.678-0.752] | 0.712 | -0.006 [-0.015, 0.005] |
| età 50-64 | 1365 | 538 | 0.739 [0.715-0.764] | 0.739 | 0.000 [-0.006, 0.006] |
| età 65-74 | 1073 | 478 | 0.683 [0.654-0.715] | 0.688 | +0.005 [-0.002, 0.014] |
| età >=75 | 1219 | 710 | 0.734 [0.710-0.762] | 0.722 | -0.012 [-0.019, -0.004] |
| emergency | 1688 | 638 | 0.765 [0.740-0.786] | 0.768 | +0.003 [-0.003, 0.010] |
| inpatient | 1903 | 1013 | 0.707 [0.682-0.731] | 0.703 | -0.004 [-0.010, 0.003] |
| outpatient | 858 | 240 | 0.741 [0.701-0.776] | 0.741 | 0.000 [-0.015, 0.014] |
| procedural | 177 | 99 | 0.676 [0.595-0.742] | 0.663 | -0.013 [-0.068, 0.039] |
| etnia asian | 134 | 62 | 0.698 [0.611-0.785] | 0.658 | -0.040 [-0.128, 0.042] |
| black | 728 | 325 | 0.748 [0.711-0.781] | 0.735 | -0.013 [-0.027, 0.002] |
| hispanic | 1351 | 468 | 0.743 [0.713-0.771] | 0.745 | +0.002 [-0.007, 0.011] |
| other | 380 | 196 | 0.727 [0.681-0.779] | 0.708 | -0.019 [-0.042, 0.003] |
| unknown | 648 | 310 | 0.779 [0.750-0.813] | 0.762 | -0.018 [-0.035, 0.000] |
| white | 1385 | 629 | 0.738 [0.713-0.763] | 0.739 | +0.001 [-0.007, 0.010] |

Il modello locale non batte mai il globale in modo significativo (nessuna differenza con IC inferiore >0; 2 su 19 strati con IC interamente <0 per il composito, più 'unknown' al limite (IC superiore 0.000)). Per stenosi aortica e MR (stessa analisi, `b_strata.csv`): 0 differenze positive significative su 28 confronti validi, 6 significativamente negative (es. MR QRS>=120 -0.042 [-0.071,-0.009]; MR etnia white -0.058 [-0.091,-0.025]). I modelli locali con meno dati sono peggiori: conviene il modello globale.
Equità: per il composito l'AUROC globale per etnia varia tra 0.698 e 0.779 e per setting tra 0.676 (procedural) e 0.765; le differenze tra gruppi non sono però dimostrate (IC ampi per asian e procedural). Per età: la stenosi aortica scende a 0.634 [0.542-0.734] (65-74) e 0.717 [0.676-0.761] (>=75) contro 0.845 [0.820-0.876] per QRS <100: la discriminazione sembra guidata dall'età (limite del segnale tabulare). Setting e distribuzione per etnia differiscono tra train e val (shift): vedi Problemi aperti.

**(c) Sottogruppi: 1950 regole provate in train (OOF), 15 verificate su validation (`results/P2/c_rules_val.csv`). Nessuna credibile.**
Composito (globale val 0.746): la migliore in train (età<52 AND PR>=174, margine OOF +0.065) in validation ha AUROC 0.762 [0.694-0.837] su n=187 (margine +0.015, sotto 0.02 e IC inferiore 0.694 < 0.746). La regola 2 (età<63 AND PR>=174, n=425) dà 0.779 [0.731-0.832], margine +0.033 ma IC inferiore 0.731 < 0.746: NON credibile. Le altre 3 hanno margine ≤0.011.
Stenosi aortica (globale val 0.844): ventricular_rate>=95 AND qt_corrected<427 in val ha AUROC 0.950 [0.904-0.986] ma n=189 con soli 8 positivi (<25): non valutabile con affidabilità, non credibile; le altre 4 regole hanno 9-18 positivi (<25).
Mitral regurgitation (globale val 0.767): le 5 regole in val hanno AUROC sottogruppo ≤0.797 e IC larghi; nessuna con margine≥0.02 e IC inferiore sopra il globale (la migliore, qrs>=90 AND qt<427, 0.797 [0.689-0.881], n=485, 23 positivi).
Il modello locale è peggiore o uguale al globale in tutte le 15 regole (diff tra -0.052 e +0.024, mai con IC inferiore >0). Le regole che crollano in validation (es. MR: qrs>=102 AND qt<427, margine OOF +0.041 in train → -0.097 in val) sono riportate come tali: nessun segnale di sottogruppo replicato.

## Decisioni e motivazioni
- Target per (b)/(c): composito + i 2 promettenti con AUROC più alto (la scelta usa il validation, ottimistico). Limitato a 3 per contenere la multiplicità (1950 regole).
- Candidato di un sottogruppo = modello globale hist_gb ristretto dalla regola (pre-specificato, non scelto in base a locale vs globale): il confronto equo è in `c_rules_val.csv`.
- Nessun candidato salvato in `results/P2/candidates/`: nessun sottogruppo soddisfa tutti i criteri (è un risultato).
- Regole su location/etnia escluse: non sono feature in ingresso.
- Alternativa scartata: soglie per ogni feature a più quantili (più regole, più multiplicità).

## Configurazioni provate
- 1950 regole provate sul solo train (OOF); 15 regole verificate su validation.
- Registrazioni `log-exp` per P2: E0002 (smoke test, 148 valutazioni val) e E0003 (run completo, 160 valutazioni val = 36 modelli-target + 2 per ciascuno dei 47 confronti locale/globale tra strati e regole valide, calcolate dallo script), totale 308. Gate: N_REGOLE_PROVATE=1950 > 500 senza replica indipendente: i risultati di sottogruppo andrebbero comunque trattati con cautela (qui non ce ne sono di credibili).

## Artefatti
- `experiments/P2/conditions.py`; `results/P2/a_conditions.csv`, `a_skipped.csv`, `b_strata.csv`, `c_rules_val.csv`, `summary.json`; modelli e OOF in `results/P2/*.joblib`, `oof_*.npy` (file a livello di paziente, non esportare); smoke test in `results/P2/debug/`. Nessuna cartella in `results/P2/candidates/`.

## Problemi aperti
- Shift train/val: prevalenza composito 0.524 vs 0.430 e composizione per setting diversa (inpatient 41% in val vs 48% in train; emergency 36% vs 31%): le AUROC non risentono direttamente della prevalenza ma la calibrazione sì (non valutata).
- Selezione di target e del miglior modello usa il validation: AUROC leggermente ottimistiche.
- Condizioni rare (regurgitazione aortica/polmonare, versamento): pochi positivi, IC ampi, nessun risultato affidabile.
- Per le stenosi aortica regole con AUROC>0.9 hanno <25 positivi: servirebbe più dati per verificarle.
- Gli IC bootstrap trattano gli ECG come indipendenti; in val c'è 1 ECG per paziente (ok), in train più ECG per paziente (gestito con GroupKFold per l'OOF).

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- TARGET_PROMETTENTI: [aortic_stenosis (AUROC val 0.844 [0.819-0.868]), mitral_regurgitation (0.770 [0.741-0.793]), lvef_lte_45 (0.764 [0.748-0.782]), rv_systolic_dysfunction (0.764 [0.737-0.787]), tricuspid_regurgitation (0.756 [0.730-0.783]), pasp_gte_45, tr_max_gte_32, lvwt_gte_13]
- Non promettenti: pericardial_effusion (0.678, IC inf 0.604), aortic_regurgitation (0.674, IC inf 0.592), pulmonary_regurgitation (21 pos val, 0.613)
- SOTTOGRUPPI_CREDIBILI: [] (nessuno; 15 regole verificate su val, nessuna con margine >=0.02 e IC inferiore sopra il globale con >=25 positivi)
- N_REGOLE_PROVATE: 1950 (train, OOF); 15 verificate su validation; configurazioni totali P2 su validation registrate: 308
- Modello locale vs globale (composito, 19 strati): nessuna differenza positiva significativa; il globale è meglio o uguale
- Composito AUROC val globale 0.746 [0.733-0.759]; 0 candidati salvati
RACCOMANDAZIONE:
- Gate P2: cluster_targets = composito + aortic_stenosis + mitral_regurgitation (i 2 promettenti con AUROC più alto) o + lvef_lte_45; nessun sottogruppo credibile da portare in P6.
- Segnalare: N_REGOLE_PROVATE > 500 (gate), nessuna replica indipendente.
