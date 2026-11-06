# Report R1-P4 · review-waveform
- Agente: waveform-modeler
- Data: 2026-11-06 15:40
- Stato: OK

## Obiettivo
Rispondere alla revisione P6 su P4: (a) la CNN candidata funziona in modo omogeneo nei sottogruppi di validation (etnia, setting di ricovero, sesso, età) e batte il modello tabulare `hist_gb` anche lì; (b) `candidate.predict` riproduce le predizioni salvate. Nessun riaddestramento. Ricordiamo che SHD non coincide con scompenso cardiaco.

## Input letti
- Handoff P0, P1, P4 (`python3 tools/state.py brief`); nel PLAYBOOK non esiste una sezione «R1-P4» testuale, ho seguito la descrizione della fase.
- `results/P4/val_pred_emb.npz` (predizioni CNN su validation, 4626 righe, 12 target; colonna 0 = composito, verificato l'allineamento con `row_in_split`).
- `results/P1/candidates/global_best` (hist_gb su 7 feature ECG, per le predizioni tabulari su val).
- `results/P4/candidates/cnn_multilabel` (candidate.py, pesi, definizione rete).

## Cosa è stato fatto
Script `experiments/R1-P4/strat.py` (nessuno smoke test separato: non si addestra nulla, si usano predizioni già salvate e un modello già addestrato; esecuzione diretta sull'intera validation).
1. AUROC del composito (`shd_moderate_or_greater_flag`) per ogni livello di `race_ethnicity`, `location_setting`, `sex` (0 = femmina, 1 = maschio), `age_bin` (<50, 50-64, 65-74, >=75), per CNN e hist_gb, con IC95% bootstrap (1000 ricampionamenti).
2. Differenza CNN − hist_gb con IC95% **appaiato** (stesso ricampionamento di righe per entrambi i modelli). Una riga per paziente in val, quindi il bootstrap per riga equivale a quello per paziente.
3. `candidate.predict` chiamato su 500 righe di validation (campione casuale, seed 42) e confrontato con la predizione salvata.

## Risultati
Tutti i valori sono su **validation** (globale: CNN 0.823, hist_gb 0.746). Soglia di inclusione: almeno 10 positivi e 10 negativi (tutti gli strati la superano).

| Variabile | Livello | n | positivi | AUROC CNN [IC95%] | AUROC hist_gb [IC95%] | Differenza appaiata [IC95%] |
|---|---|---|---|---|---|---|
| etnia | asian | 134 | 62 | 0.788 [0.706-0.859] | 0.698 [0.611-0.782] | +0.090 [0.007, 0.170] |
| etnia | black | 728 | 325 | 0.838 [0.810-0.869] | 0.748 [0.711-0.784] | +0.090 [0.055, 0.126] |
| etnia | hispanic | 1351 | 468 | 0.819 [0.795-0.843] | 0.743 [0.714-0.770] | +0.076 [0.048, 0.102] |
| etnia | other | 380 | 196 | 0.811 [0.771-0.851] | 0.727 [0.676-0.775] | +0.084 [0.034, 0.138] |
| etnia | unknown | 648 | 310 | 0.840 [0.810-0.870] | 0.779 [0.743-0.813] | +0.061 [0.030, 0.097] |
| etnia | white | 1385 | 629 | 0.810 [0.785-0.833] | 0.738 [0.711-0.764] | +0.073 [0.049, 0.097] |
| setting | emergency | 1688 | 638 | 0.833 [0.815-0.853] | 0.765 [0.742-0.788] | +0.069 [0.044, 0.092] |
| setting | inpatient | 1903 | 1013 | 0.803 [0.784-0.823] | 0.707 [0.683-0.730] | +0.096 [0.073, 0.118] |
| setting | outpatient | 858 | 240 | 0.808 [0.775-0.838] | 0.741 [0.705-0.778] | +0.066 [0.030, 0.105] |
| setting | procedural | 177 | 99 | 0.748 [0.677-0.818] | 0.676 [0.591-0.752] | +0.072 [-0.002, 0.150] |
| sesso | femmina (0) | 2356 | 869 | 0.817 [0.799-0.835] | 0.739 [0.719-0.759] | +0.078 [0.059, 0.099] |
| sesso | maschio (1) | 2270 | 1121 | 0.822 [0.804-0.837] | 0.736 [0.716-0.755] | +0.086 [0.067, 0.105] |
| età | <50 | 969 | 264 | 0.831 [0.801-0.861] | 0.717 [0.679-0.753] | +0.114 [0.081, 0.148] |
| età | 50-64 | 1365 | 538 | 0.826 [0.805-0.850] | 0.739 [0.713-0.767] | +0.087 [0.058, 0.114] |
| età | 65-74 | 1073 | 478 | 0.802 [0.775-0.828] | 0.683 [0.651-0.715] | +0.120 [0.088, 0.151] |
| età | >=75 | 1219 | 710 | 0.788 [0.762-0.813] | 0.734 [0.706-0.761] | +0.054 [0.024, 0.083] |

Lettura:
- La CNN è sopra hist_gb in tutti i 16 strati; l'IC appaiato esclude 0 in 15/16 (unica eccezione: setting procedurale, n=177, IC fino a −0.002).
- Il vantaggio varia da +0.054 (età >=75) a +0.120 (65-74); il minimo AUROC CNN è 0.748 (procedurale, IC largo) e 0.788 (>=75). Gli IC dei livelli sono ampi per asian (n=134) e procedural (n=177): differenze fra strati vicini non sono dimostrate (gli IC si sovrappongono quasi ovunque) e non sono confronti formali fra strati.
- Nessuna prova di prestazioni molto peggiori in un gruppo etnico, di sesso o di setting; la CNN tende a peggiorare con l'età (0.831 → 0.788) ed è più bassa in ricovero (0.803) e procedurale (0.748). Sono osservazioni descrittive.

**`candidate.predict` su 500 righe di validation** (`results/R1-P4/candidate_check.json`):
- Colonne restituite: `score`, `in_scope` (dtype score float32); `in_scope` True per tutte le 500 righe.
- Indice: identico a quello del DataFrame in ingresso (True).
- max|score − predizione salvata| = 1.96e-4 (media 3.9e-7); range score 0.0094–0.9996. La differenza è dovuta all'autocast float16 e a batch diversi rispetto al salvataggio: trascurabile (nessun effetto sul ranking osservabile).

## Decisioni e motivazioni
- Strati con le variabili richieste, soglie di età già usate in `lib.add_groups`. Nessuna scelta di modello o di soglia basata sui risultati: analisi descrittiva.
- IC appaiato per il confronto, perché i due modelli sono valutati sulle stesse righe (più preciso dei due IC separati).
- Non riaddestrato nulla (richiesta della fase).

## Configurazioni provate
1 valutazione descrittiva su validation (registrata come E0020 con `log-exp --uses-val --n-configs 1`). Nessun modello scelto guardando il validation; i numeri restano comunque calcolati su un validation già usato in P1-P4, quindi i confronti tra strati sono esplorativi (16 strati × 2 modelli, senza correzione per confronti multipli).

## Artefatti
- `experiments/R1-P4/strat.py`
- `results/R1-P4/val_stratified.csv` (aggregati per strato)
- `results/R1-P4/candidate_check.json`

## Problemi aperti
- Un solo seed e una sola configurazione della CNN (da P4); valutazione su validation già consultato.
- Le categorie etniche sono quelle del dataset (`unknown`, `other` non interpretabili); piccola numerosità per asian e procedural.
- Il confronto con hist_gb sul solo composito; non stratificati gli altri target.
- Il test finale (P7) resta il vero banco di prova; qui nessun dato di test è stato aperto.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- CNN > hist_gb in 16/16 strati (etnia, setting, sesso, età); IC appaiato >0 in 15/16; eccezione procedural n=177: +0.072 [-0.002, 0.150]
- AUROC CNN più bassa: procedural 0.748 [0.677-0.818], età >=75 0.788 [0.762-0.813]; più alta: unknown 0.840, black 0.838 (globale 0.823)
- Vantaggio appaiato da +0.054 [0.024, 0.083] (>=75) a +0.120 [0.088, 0.151] (65-74)
- Sesso: femmina 0.817 [0.799-0.835], maschio 0.822 [0.804-0.837]
- candidate.predict (500 righe val): colonne score,in_scope; indice identico; max|score-salvata| = 1.96e-4
- Configurazioni su val: +1 (E0020)
RACCOMANDAZIONE:
- P6: nessuna evidenza di fallimento in un sottogruppo; il candidato cnn_multilabel può essere ammesso con le cautele già note (un seed, val consultato); confermare in P7.
