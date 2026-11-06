# Report Q1 · cnn-cv
- Agente: cnn-cv-analyst
- Data: 2026-11-06 18:40
- Stato: OK

## Obiettivo
Dare alla CNN sulle forme d'onda una stima solida, ottenuta con cross-validation (CV) a 5 fold per paziente e 3 addestramenti con seed diversi, mentre la pipeline 1 aveva un solo addestramento. Confrontarla con il modello tabulare (7 misure ECG) sulle stesse righe e fold, e controllare quanto contano le righe senza LVEF. Ricordiamo che SHD (cardiopatia strutturale) non coincide con lo scompenso cardiaco.

## Input letti
- `python3 tools/state2.py brief` e Handoff di Q0: LEAD_ORDER_OK=yes, N_DEV=77101 ECG, popolazione clean 70130 ECG (91.0%), 9.5 s/epoca su GPU, EST_MIN_Q1=30 min.
- `agentic/PLAYBOOK2.md` (sezione Q1 e regole speciali).
- Riferimenti pipeline 1 (dall'handoff Q0 e dal playbook): CNN validation P4 0.823; SHD test P7 CNN 0.830 [0.820-0.841], tabulare 0.762 [0.750-0.774].

## Cosa è stato fatto
1. Run principale: `tools/cnn_cv.py run --out results/Q1/main --seeds 0,1,2 --epochs 12` (tutte le 12 derivazioni, popolazione clean). Sono 15 addestramenti (5 fold × 3 seed), completati in 5 invocazioni con `--time-budget-min 7`. Poi `summarize`. L'AUROC è calcolata sulle predizioni fuori-fold (ogni paziente è predetto da una rete che non lo ha visto), con la media delle probabilità dei 3 seed (ensemble) e IC95% bootstrap per paziente. Il modello tabulare è un hist_gb sulle 7 misure, con le stesse fold.
2. Sensibilità alle righe senza LVEF: `--population all --seeds 0 --out results/Q1/all_rows` (5 addestramenti, anche sulle righe senza LVEF). Confronto con il seed 0 del run principale sulle sole righe clean, con differenza appaiata e IC per paziente (`experiments/Q1/sens.py`).
3. Esplorazione di varianti (epoche, lr, reg-weight): **non eseguita**. È opzionale nel playbook e non c'era motivo di cercare oltre, quindi la configurazione principale resta quella finale.
4. Smoke test con `debug_n`: non rieseguito. Lo strumento `cnn_cv.py` era già stato provato in Q0 (benchmark di 2 epoche sul pool intero).

## Risultati
Valori in CV per paziente, popolazione clean (n=70130), ensemble di 3 seed. «diff» = CNN − tabulare, con IC appaiato per paziente. Le classi sono indicative (buona: AUROC ≥ 0.80 con IC inferiore ≥ 0.75; moderata 0.70-0.80; debole < 0.70).

| Etichetta | n positivi | CNN AUROC [IC95%] | Tabulare [IC95%] | diff [IC95%] | SD tra seed | Classe |
|---|---|---|---|---|---|---|
| SHD composito | 39948 | 0.815 [0.809-0.822] | 0.712 [0.705-0.721] | +0.103 [0.096-0.109] | 0.000 | buona |
| LVEF ≤ 45% | 17828 | 0.869 [0.862-0.876] | 0.720 [0.710-0.730] | +0.149 [0.141-0.159] | 0.001 | buona |
| Spessore parete ≥ 13 mm | 18544 | 0.739 [0.730-0.747] | 0.652 [0.642-0.663] | +0.087 [0.076-0.096] | 0.001 | moderata |
| Stenosi aortica | 3171 | 0.811 [0.794-0.826] | 0.841 [0.825-0.857] | −0.030 [−0.045 −0.014] | 0.001 | buona |
| Insufficienza aortica | 940 | 0.692 [0.652-0.732] | 0.602 [0.559-0.640] | +0.089 [0.045-0.126] | 0.002 | debole |
| Insufficienza mitralica | 6419 | 0.789 [0.775-0.802] | 0.700 [0.681-0.716] | +0.089 [0.077-0.104] | 0.000 | moderata |
| Insufficienza tricuspidale | 8012 | 0.783 [0.772-0.795] | 0.694 [0.679-0.711] | +0.089 [0.073-0.103] | 0.001 | moderata |
| Insufficienza polmonare | 624 | 0.767 [0.737-0.797] | 0.598 [0.535-0.666] | +0.169 [0.114-0.225] | 0.003 | moderata |
| Disfunzione sistolica VD | 9965 | 0.844 [0.834-0.853] | 0.718 [0.705-0.730] | +0.126 [0.116-0.137] | 0.000 | buona |
| Versamento pericardico mod./grande | 2131 | 0.714 [0.685-0.738] | 0.624 [0.590-0.652] | +0.090 [0.062-0.122] | 0.005 | moderata |
| PASP ≥ 45 | 14308 | 0.748 [0.737-0.758] | 0.648 [0.637-0.660] | +0.099 [0.089-0.111] | 0.002 | moderata |
| TR max ≥ 32 | 7759 | 0.750 [0.736-0.762] | 0.641 [0.627-0.654] | +0.108 [0.096-0.124] | 0.001 | moderata |

- **Variabilità tra addestramenti**: per l'AUROC di SHD i 3 seed danno 0.808-0.809 (SD < 0.001; AUROC di un singolo seed, non dell'ensemble). La variabilità dovuta al seed è quindi trascurabile rispetto all'IC per paziente (±0.007). L'ensemble sale di circa +0.007 rispetto al singolo seed (0.815 contro 0.808). Non ho misurato la variabilità tra fold.
- **LVEF continua** (n=70130): CNN MAE 7.48 punti di EF, R² 0.482. Tabulare MAE 10.45, R² 0.156. IC non calcolati per questi due numeri (limite).
- **Confronto con la pipeline 1**: l'AUROC SHD in CV clean (0.815) è in linea con il validation di P4 (0.823) e con il test P7 (0.830), anche se le popolazioni non sono identiche. Il tabulare in CV clean (0.712) è più basso di quello di P1 (test 0.762) e va letto come numero su una popolazione e un modello diversi (hist_gb, righe clean). Non dimostra un errore nel test.
- **Sensibilità alle righe senza LVEF** (seed 0, valutata sulle sole righe clean, differenza = addestramento su tutte le righe − addestramento solo clean, IC appaiato per paziente):

| Etichetta | Solo clean | Tutte le righe | Differenza [IC95%] |
|---|---|---|---|
| SHD | 0.8075 [0.801-0.814] | 0.8095 [0.803-0.817] | +0.0020 [0.0001-0.0038] |
| LVEF ≤ 45% | 0.8622 [0.855-0.869] | 0.8648 [0.858-0.872] | +0.0026 [0.0003-0.0048] |

  Le righe senza LVEF cambiano l'AUROC di circa 0.002-0.003, un effetto minuscolo (il limite inferiore dell'IC è appena sopra 0). Il risultato principale non dipende da quelle righe. Valutate su *tutte* le righe, le AUROC salgono (SHD 0.829 [0.823-0.835]), perché le righe anomale con etichette tutte a 0 sono facili da separare. Questo non è un guadagno reale e non va citato come prestazione.

## Decisioni e motivazioni
- `CNN_DIR` = `results/Q1/main`; nessuna variante provata, quindi nessuna sostituzione.
- Popolazione clean come risultato principale (regola 4 del playbook). La popolazione all è una sensibilità, e le sue AUROC più alte non vanno usate come risultato.
- Con 0.000-0.005 di SD tra seed, 3 seed bastano per la stabilità, ma l'incertezza residua è quella del campione di pazienti (IC).
- Non sono nominate le derivazioni toraciche (non verificabili).

## Configurazioni provate
2 configurazioni, entrambe con uso del validation (la CV usa train+validation): E0023 (main, 3 seed) ed E0024 (sensibilità population all, seed 0). Nessuna selezione tra alternative. Le AUROC sono quindi stime in CV non ottimistiche per scelta di iperparametri, ma i dati di sviluppo sono quelli già visti in P1-P5.

## Artefatti
- `results/Q1/main/` (oof per seed/fold, embedding, pesi, `summary.json`, `metrics_clean.csv`, `metrics_all.csv`, `oof_ensemble.npy`, `tabular_oof.npz`)
- `results/Q1/all_rows/` (stessa struttura, 1 seed)
- `results/Q1/sensitivity.json`, `experiments/Q1/sens.py`
- `reports/Q1_cnn-cv.progress.md`

## Problemi aperti
- IC non calcolati per MAE e R² della LVEF continua, né per la variabilità tra fold.
- Le righe senza LVEF restano anomale (tutte le etichette a 0): il motivo non è recuperabile con questi dati.
- Il tabulare CV clean (0.712) è più basso del riferimento P1 (0.762 sul test): da non sovrainterpretare.
- Le classi sono indicative: la classe definitiva si ha solo sul test (Q8).
- La stenosi aortica è l'unica etichetta in cui il tabulare è migliore (differenza −0.030 [−0.045 −0.014]).
- Esplorazione di varianti non eseguita.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- CNN_DIR: results/Q1/main
- AUROC_SHD_CLEAN: 0.815 [0.809-0.822]; SEED_SD: <0.001 (seed singoli 0.808-0.809)
- TAB_AUROC_SHD_CLEAN: 0.712 [0.705-0.721]; DIFF_SHD: +0.103 [0.096-0.109]
- TARGETS_GOOD: [shd, lvef_lte_45 (0.869), rv_systolic_dysfunction (0.844), aortic_stenosis (0.811)]
- TARGETS_MODERATE: [lvwt_gte_13, mitral_regurgitation, tricuspid_regurgitation, pulmonary_regurgitation, pericardial_effusion_moderate_large, pasp_gte_45, tr_max_gte_32]; TARGETS_WEAK: [aortic_regurgitation 0.692 [0.652-0.732]]
- LVEF_REG: CNN MAE 7.48 / R2 0.482 contro tabulare MAE 10.45 / R2 0.156
- SENSIBILITA_RIGHE_SENZA_LVEF: delta SHD +0.0020 [0.0001-0.0038]; LVEF≤45% +0.0026 [0.0003-0.0048]
- CONFIG_FINALE: epochs 12, lr 0.002, reg_weight 0.5, batch 128, leads all (12), popolazione clean
RACCOMANDAZIONE:
- Le fasi successive usano results/Q1/main. Nessuna variante ha sostituito la principale.
- Aortic stenosis: il tabulare è migliore della CNN. Tenerne conto nei candidati (Q6).
