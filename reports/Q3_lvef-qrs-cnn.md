# Report Q3 · lvef-qrs-cnn
- Agente: lvef-qrs-cnn-analyst
- Data: 2026-11-06 20:30
- Stato: OK

## Obiettivo
I medici ipotizzano che la durata del QRS (il tratto dell'ECG che riflette l'attivazione dei ventricoli) spieghi perché, a parità di altre misure, la frazione di eiezione (LVEF) cambia. In Q3 verifichiamo l'ipotesi con un predittore molto migliore del tabulare (la CNN sulle forme d'onda): il QRS aggiunge informazione su ciò che la CNN non spiega? Nota: SHD e LVEF bassa non coincidono con lo scompenso cardiaco.

## Input letti
- Handoff Q1: `CNN_DIR = results/Q1/main`; CNN LVEF MAE 7.48 / R² 0.482 contro tabulare MAE 10.45 / R² 0.156.
- `results/Q1/main/oof_ensemble.npy` (media di 3 seed, predizioni fuori-fold; colonna 12 = LVEF stimata) e `results/Q1/main/tabular_oof.npz` (modello tabulare sulle 7 misure, stesse fold).
- `pipe2_lib.load_dev()` per QRS, età, sesso, frequenza, PR, QTc, LVEF vera e fold per paziente. Nessun nuovo addestramento. Test set non toccato.

## Cosa è stato fatto
Script unico `experiments/Q3/qrs_cnn.py` (verifica di caricamento con `debug`, poi esecuzione completa). Popolazione *clean* (LVEF nota): 70 130 ECG, 17 828 con LVEF ≤ 45% (25,4%); il QRS è presente in tutti. Tutti gli IC95% sono bootstrap per paziente (500 ripetizioni).
- a) AUROC di LVEF ≤ 45% per fascia di QRS (<100, 100-119, ≥120 ms): CNN e tabulare, differenza appaiata, confronto della CNN tra fasce.
- b) Regressione logistica `y ~ logit(CNN)` contro `y ~ logit(CNN) + QRS + QRS×logit(CNN)`, in CV con le stesse 5 fold per paziente (QRS standardizzato, troncato ai percentili 1-99).
- c) Residuo = LVEF vera − LVEF stimata dalla CNN: correlazione di Spearman col QRS, correlazione parziale (rango, dopo età, sesso, frequenza, PR, QTc; n = 62 206 per i dati completi), ΔR² in CV (regressione lineare `LVEF ~ CNN` contro `+ QRS + QRS×CNN`).
- d) Coppie discordanti: un ECG per paziente (scelto a caso, seed 42; 27 749 pazienti), accoppiamento senza ripetizioni di pazienti con LVEF CNN che differisce < 1 punto e LVEF vera che differisce ≥ 10 punti.
- Numero di confronti: **8** (a: 3, b: 2, c: 2, d: 1), Holm applicato entro ogni famiglia. Il confronto di eterogeneità tra fasce (2 differenze) è descrittivo e fuori da questi 8.

## Risultati
Tutti i valori sono su CV per paziente (pool sviluppo, popolazione clean), non sul test.

**a) AUROC LVEF ≤ 45% per fascia di QRS**

| Fascia QRS | n | positivi | CNN AUROC [IC95%] | Tabulare AUROC [IC95%] | CNN − tab [IC95%] | p Holm |
|---|---|---|---|---|---|---|
| <100 ms | 49 035 | 9 186 | 0.840 [0.829-0.851] | 0.669 [0.658-0.680] | +0.171 [0.159-0.182] | 0.006 |
| 100-119 ms | 11 441 | 4 068 | 0.867 [0.854-0.879] | 0.662 [0.643-0.679] | +0.204 [0.186-0.221] | 0.006 |
| ≥120 ms | 9 654 | 4 574 | 0.883 [0.868-0.897] | 0.683 [0.665-0.704] | +0.200 [0.180-0.219] | 0.006 |

La CNN batte il tabulare in ogni fascia. Eterogeneità della CNN: 100-119 contro <100 = +0.027 [0.012-0.041]; ≥120 contro <100 = +0.043 [0.025-0.060]. La CNN è un po' più accurata quando il QRS è largo, ma ovunque «buona» (AUROC ≥ 0.80, IC inferiore ≥ 0.75). Il tabulare resta debole in tutte le fasce (0.66-0.68): il QRS da solo non lo rende migliore.

**b) Valore aggiunto del QRS sopra la CNN (LVEF ≤ 45%, CV)**

| Metrica | Base (solo CNN) | Con QRS + interazione | Differenza [IC95%] | p Holm |
|---|---|---|---|---|
| AUROC | 0.8688 | 0.8689 | +0.0001 [−0.0005, +0.0006] | 0.79 |
| Log-loss (base − completo; >0 = migliora) | 0.3680 | 0.3678 | +0.0002 [−0.0002, +0.0007] | 0.58 |

**c) LVEF continua**

| Quantità | Valore [IC95%] | p Holm |
|---|---|---|
| Spearman residuo-QRS | −0.011 [−0.029, +0.006] | (descrittivo) |
| Correlazione parziale residuo-QRS | +0.006 [−0.012, +0.022] | 0.49 |
| R² CV, solo CNN | 0.488 | |
| ΔR² aggiungendo QRS e interazione | +0.0004 [−0.0002, +0.0010] | 0.46 |

MAE della CNN 7.48 punti di LVEF; residuo medio −0.41 (sd 10.75).

**d) Coppie discordanti**: 8 992 coppie di pazienti diversi, con LVEF vera media diversa di 16.5 punti e LVEF CNN diversa di 0.5 punti. Il QRS di chi ha LVEF più bassa è più lungo di **+1.5 ms [0.9-2.1]** (p Holm < 0.001). È statisticamente significativo, ma l'effetto è molto piccolo (circa 1/10 della deviazione tipica del QRS tra fasce cliniche) e, coerentemente con b e c, non migliora la predizione.

**Verdetto `QRS_MODIFICA` = non_supportata.** Regola del PLAYBOOK: (b) IC superiore del ΔAUROC = +0.0006 < 0.01 e (c) IC superiore del ΔR² = +0.0010 < 0.01. Entrambi i criteri di «non supportata» sono soddisfatti con ampio margine (l'effetto è escluso, non solo non visto).

## Decisioni e motivazioni
- Verdetto secondo la regola scritta, indipendente dal punto d (che è significativo ma trascurabile e non fa parte della regola).
- Il test (a) CNN contro tabulare è un confronto di metodo; il confronto fra fasce è descrittivo. Non si afferma che il QRS «modifica» la relazione: la CNN è meglio con QRS largo (+0.04), ma questo riguarda l'accuratezza, non una informazione extra del QRS.
- Limite del ragionamento: la CNN vede già il QRS nella forma d'onda, quindi può aver assorbito l'informazione del QRS. L'esito dice che il QRS misurato non aggiunge nulla *sopra la CNN*, non che il QRS sia irrilevante per l'LVEF.
- Alternative non fatte: modelli non lineari per l'interazione (oltre il PLAYBOOK).

## Configurazioni provate
8 confronti (3+2+2+1), registrati con `log-exp` (Q3: 8 configurazioni, `--uses-val`, perché la CV include il validation). Nessuna scelta di iperparametri: le predizioni sono quelle di Q1. Più 2 differenze di eterogeneità descrittive (non nei 8).

## Artefatti
- `experiments/Q3/qrs_cnn.py`
- `results/Q3/q3_results.json` (tutti i numeri sopra)
- `reports/Q3_lvef-qrs-cnn.progress.md`

## Problemi aperti
- Un solo centro, solo pazienti con eco; popolazione clean (le righe senza LVEF escluse, motivo non chiarito).
- QRS usato come durata lineare standardizzata: effetti non lineari (es. blocco di branca) non esclusi; la fascia ≥120 è però analizzata a parte in (a).
- Il p di bootstrap ha un minimo di circa 0.002 (500 ripetizioni): i p Holm «0.006» sono un limite inferiore di risoluzione.
- Le OOF sono la media di 3 seed: il regressore di (b)/(c) è stato adattato su predizioni fuori-fold di altre fold, quindi nessun leakage del singolo paziente.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- QRS_MODIFICA: non_supportata
- AUROC_PER_FASCIA (LVEF ≤ 45%, CNN / tab): <100 0.840/0.669; 100-119 0.867/0.662; ≥120 0.883/0.683 (CNN meglio ovunque, +0.17/+0.20/+0.20)
- DELTA_AUROC_QRS: +0.0001 [−0.0005, +0.0006] (p Holm 0.79)
- DELTA_R2_QRS: +0.0004 [−0.0002, +0.0010] (p Holm 0.46); correlazione parziale residuo-QRS +0.006 [−0.012, +0.022]
- DIFF_QRS_COPPIE: +1.5 ms [0.9, 2.1] su 8 992 coppie (significativa ma trascurabile)
- N_CONFRONTI: 8 (Holm per famiglia)
RACCOMANDAZIONE:
- In Q6/Q9 riportare l'ipotesi QRS come non supportata sopra la CNN (effetto escluso oltre ±0.001 di AUROC), con la cautela che la CNN vede già il QRS.
- Non servono candidati nuovi da Q3.
