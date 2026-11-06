# Report P3 · lvef-qrs
- Agente: lvef-qrs-analyst
- Data: 2026-11-06 15:00
- Stato: OK

## Obiettivo
Stabilire quanto bene le 7 feature ECG predicono la frazione di eiezione (`lvef_value`, e `lvef_lte_45_flag`) e se i dati supportano l'ipotesi dei medici che `qrs_duration` (QRS largo, ad esempio per blocco di branca sinistra) modifichi la relazione tra feature e LVEF. Promemoria: lo studio riguarda la cardiopatia strutturale (SHD), che NON coincide con lo scompenso cardiaco.

## Input letti
- Handoff P0 (integrità OK, lvef_value mancante 8.9% in train), P1 (segnale tabulare presente), P2 (lvef_lte_45 AUROC val 0.764 [0.748-0.782]).
- `agentic/PLAYBOOK.md` sezione P3, `legacy_notebooks/02_LVEF_e_ruolo_del_QRS.ipynb` (ricetta).
- Dati solo tramite `lib.load_data` (test chiuso, mai toccato). Input dei modelli: solo `ECG_FEATURES`.

## Cosa è stato fatto
Script unico `experiments/P3/lvef_qrs.py` (con cache dei risultati in `results/P3/`). Smoke test con `debug_n=2000` (risultati in `results/P3/debug/`, registrato con log-exp), poi run completo.
Righe con LVEF mancante escluse: train 66018 ECG su 72475 (23637 pazienti), validation 4112 su 4626 (1 ECG per paziente). Per la classificazione si usano le righe con `lvef_lte_45_flag` disponibile (val n=4626, 866 positivi).
1. Regressione (ridge, random forest, hist_gb, MLP) con MAE e R² contro la baseline "media del train"; classificazione a 5 modelli.
2. Test (i) stratificazione: hist_gb locale (profondità 3) per fascia QRS (<100, 100-119, >=120) vs hist_gb globale sugli stessi pazienti di validation; bootstrap (500 ricampionamenti) per l'IC della differenza.
3. Test (ii) interazione formale, solo sul train: Ridge additivo (7 feature + indicatori di mancanza) vs Ridge con 6 prodotti `qrs × feature`, con QRS continuo e con QRS>=120 ms. Test F "naive" (ignora che lo stesso paziente ha più ECG, quindi troppo ottimista), test di Wald con errori standard robusti raggruppati per paziente (implementato a mano, statsmodels non è installato), e confronto in CV a 5 fold raggruppata per paziente con bootstrap per paziente sulla differenza di errore quadratico medio (MSE).
4. Test (iii) coppie: per ogni paziente il vicino più prossimo (altro paziente) sulle feature `MATCH_ON` standardizzate (sesso pesato x10 per quasi-esattezza); tenuto il 20% di coppie più vicine; |ΔLVEF|>=15 (grande) vs <5 (piccolo). Misure: differenza delle mediane di |ΔQRS| con IC bootstrap e Mann-Whitney; segno di ΔQRS nel membro con LVEF più bassa (mediana, frazione, Wilcoxon). Train: 1 ECG casuale per paziente (indipendenza); replica su validation. Quattro insiemi MATCH_ON.

## Risultati
**(a) Regressione, validation (n=4112 con LVEF).** Baseline media del train: MAE 11.19 [10.94-11.47], R² -0.024.

| modello | MAE [IC95%] | R² [IC95%] |
|---|---|---|
| ridge | 9.73 [9.49-10.00] | 0.146 [0.121-0.174] |
| random forest | 9.54 [9.30-9.82] | 0.152 [0.123-0.183] |
| hist_gb | 9.54 [9.31-9.79] | 0.163 [0.136-0.191] |
| MLP | 9.57 [9.35-9.84] | 0.158 [0.127-0.186] |

Guadagno di MAE di hist_gb sulla baseline: 1.65 punti LVEF [1.50-1.80]. Il modello batte la baseline in modo certo, ma spiega solo ~16% della varianza: l'errore medio resta di ~9.5 punti percentuali (accuratezza individuale debole).

**(a) Classificazione LVEF<=45%, validation (n=4626, 866 positivi, prevalenza 0.187; train 0.234).**

| modello | AUROC [IC95%] | AUPRC |
|---|---|---|
| hist_gb | 0.764 [0.748-0.782] | 0.443 |
| random forest | 0.762 [0.745-0.778] | 0.443 |
| lasso_logreg | 0.759 [0.742-0.774] | 0.425 |
| logreg | 0.758 [0.742-0.774] | 0.425 |
| MLP | 0.756 [0.738-0.773] | 0.440 |

Accuratezza "moderata" (0.70-0.80); la differenza fra modelli è dentro il rumore. Il miglior modello è scelto sul validation (lieve ottimismo).

**(b-i) Stratificazione per QRS, validation (hist_gb locale meno globale; ΔMAE<0 e ΔR²>0 = locale meglio).**

| fascia QRS | n val | LVEF media | MAE glob | MAE loc | ΔMAE [IC] | R² glob | R² loc | ΔR² [IC] |
|---|---|---|---|---|---|---|---|---|
| <100 | 2953 | 55.5 | 8.18 | 8.16 | -0.012 [-0.043, 0.014] | 0.048 | 0.055 | +0.007 [0.001, 0.013] |
| 100-119 | 612 | 49.1 | 12.33 | 12.25 | -0.079 [-0.225, 0.049] | 0.103 | 0.107 | +0.004 [-0.014, 0.023] |
| >=120 | 547 | 44.6 | 13.77 | 13.64 | -0.129 [-0.274, 0.017] | 0.147 | 0.156 | +0.009 [-0.007, 0.024] |

Il modello locale non è meglio in modo rilevante: tutti gli IC della ΔMAE includono 0, un solo ΔR² ha IC fuori da 0 ma vale 0.007 (trascurabile). Nota: il LVEF medio scende con il QRS (55.5 → 44.6) e il QRS è tra le feature, quindi le fasce differiscono anche per livello di LVEF e di errore. Test (i): NON concorde con l'ipotesi.

**(b-ii) Interazione formale, train (n=66018 ECG, 23637 pazienti; 6 termini `qrs × feature`).**

| codifica QRS | F naive (p) | Wald raggruppato per paziente (p) | CV raggruppata: ΔMSE additivo-interazione [IC95%] | R² CV additivo → interazione |
|---|---|---|---|---|
| continuo | 25.9 (p<1e-4) | 7.60 (p<1e-4) | +0.28 [-0.03, 0.56] | 0.1400 → 0.1413 |
| QRS>=120 | 18.4 (p<1e-4) | 2.96 (p=0.0069) | +0.08 [-0.19, 0.37] | 0.1400 → 0.1404 |

Termini singoli (QRS continuo, Wald per paziente): qrs×qt_corrected -0.31 (p=0.0008), qrs×atrial_rate -0.30 (p=0.0020), qrs×ventricular_rate -0.24 (p=0.044); gli altri non significativi (qrs principale: -2.86 punti LVEF per 1 DS di QRS). Con 6 termini testati e molti ECG correlati, le p dei singoli coefficienti non sono corrette per la multiplicità. Il test formale rileva dunque una interazione statisticamente non nulla, ma la CV raggruppata (la misura più vicina all'uso reale) mostra un guadagno di errore di ~0.1-0.15% con IC che include 0 (e R² +0.001). Test (ii): effetto statisticamente presente ma praticamente trascurabile e non confermato dalla CV; lo conto come concordanza solo PARZIALE (non pienamente concorde).

**(b-iii, c) Coppie simili con LVEF diverso.** Train = 1 ECG per paziente; Val = replica. "Grande" = |ΔLVEF|>=15 pt, "piccolo" <5 pt. Sensibilità a 4 insiemi MATCH_ON:

| MATCH_ON | split | coppie (grandi/piccole) | mediana |ΔQRS| grandi vs piccole (ms) | differenza [IC95%] | p MW | ΔQRS(LVEF bassa) mediano ms [IC] | % coppie con QRS più largo nella LVEF bassa [IC] |
|---|---|---|---|---|---|---|---|---|
| A età, sesso, freq | train | 18206 (6439/4435) | 18 vs 12 | 6 [6-6] | <1e-4 | 10 [10-12] | 66.5% [65.3-67.6] |
| A | val | 1083 (371/266) | 18 vs 10 | 8 [2.5-10] | <1e-4 | 14 [10-16] | 69.0% [64.7-73.9] |
| B età, sesso | train | 23637 (8958/5054) | 18 vs 12 | 6 [6-6] | <1e-4 | 8 [8-8] | 61.8% [60.9-62.9] |
| B | val | 4109 (1465/914) | 20 vs 12 | 8 [6-8] | <1e-4 | 14 [12-16] | 71.5% [69.3-73.7] |
| C età, sesso, freq, PR, QTc | train | 4751 (1453/1263) | 14 vs 12 | 2 [2-6] | <1e-4 | 4 [4-6] | 57.8% [55.4-60.3] |
| C | val | 823 (253/217) | 14 vs 12 | 2 [-2, 4] | 0.065 | 6 [2-10] | 60.5% [54.6-66.2] |
| D età, freq, PR, QTc (senza sesso) | train | 4730 (1517/1161) | 16 vs 12 | 4 [2-6] | <1e-4 | 8 [6-8] | 61.9% [59.6-64.4] |
| D | val | 824 (238/212) | 16 vs 14 | 2 [0-6] | 0.0022 | 10 [6-12] | 64.3% [58.4-70.6] |

(La differenza tra mediane di valori interi in ms ha IC discreti, da leggere con cautela. Il bootstrap è sulle coppie: un paziente può comparire in più coppie, quindi gli IC sono un po' stretti.)
Conclusione sulla sensibilità: la direzione non cambia con nessun insieme (nelle coppie con LVEF molto diverso il QRS è più largo nel membro con LVEF più bassa, in 58-72% dei casi), ma l'ampiezza dipende molto da MATCH_ON: la differenza passa da 6-8 ms (A, B) a 2-4 ms (C, D) quando si appaiano anche PR/QTc, cioè gran parte dell'effetto è spiegata da altre feature ECG. Con l'insieme C la replica su validation ha IC che include 0 per la differenza di |ΔQRS|. Test (iii): concorde in direzione, di ampiezza modesta e dipendente dal MATCH_ON.
ATTENZIONE: questo test mostra che QRS e LVEF sono associati (coefficiente marginale del QRS: -2.9 punti LVEF per DS), NON che il QRS modifichi la relazione di altre feature con LVEF. Per questo non è un'evidenza diretta di "modifica d'effetto".

**Cautela causale.** Coppie simili sulle sole feature tabulari restano pazienti clinicamente diversi (terapie, comorbidità, cardiopatia ischemica, farmaci, dispositivi); un'associazione non è causalità e il QRS può essere un marcatore di cardiopatia più che una causa di LVEF diversa.

**Giudizio QRS_MODIFICA = inconclusiva.** Test (i) non concorde; (ii) significativo come p ma trascurabile e non confermato in CV; (iii) concorde ma misura l'associazione marginale e il suo peso dipende da MATCH_ON. Solo 1 test su 3 concorda in modo pieno con incertezza dichiarata (regola: servono >=2), quindi non "supportata". Neppure "non_supportata" in senso forte, perché c'è un'interazione statisticamente non nulla di piccola entità. Numeri: effetto di interazione ≤0.15% di errore quadratico in CV.

## Decisioni e motivazioni
- Giudizio "inconclusiva" (vedi sopra); per il gate P3 non serve P3b (non "supportata").
- Test (ii) fatto sul solo train (CV raggruppata) per non consumare il validation; i test F naive sono riportati ma non usati (pseudo-replicazione: 66018 ECG vs 23637 pazienti).
- Coppie: 1 ECG per paziente nel train, vicino in altro paziente, e replica su val. Sesso pesato per appaiare quasi esattamente.
- Candidato salvato: `lvef_lte45_global` (modello globale hist_gb su 7 feature, nessun sottogruppo). Target "promettente" secondo PLAYBOOK (n_pos val 866>=25, AUROC lo 0.748>=0.65); accuratezza solo moderata (<0.80). Nessun sottogruppo QRS è credibile (nessuna differenza locale-globale), quindi nessun candidato di sottogruppo.
- Scartato: modello additivo non lineare con QRS (non è un test di interazione).

## Configurazioni provate
17 configurazioni su validation, tutte registrate con log-exp: smoke test (1), regressione (4), classificazione (5), stratificazione (3 modelli locali), coppie su 4 insiemi MATCH_ON replicate su val (4, `--n-configs 1` ciascuno). Il test (ii) e le coppie su train non usano il validation. Il miglior modello di classificazione/regressione è scelto sul validation: AUROC/R² leggermente ottimistici.

## Artefatti
- Script: `/home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/experiments/P3/lvef_qrs.py`
- Risultati: `results/P3/a_regression.csv`, `a_classification.csv`, `b1_stratification.csv`, `b2_interaction.csv`, `b2_interaction_terms.csv`, `b3_pairs.csv`, `summary.json`; smoke in `results/P3/debug/`
- Candidato: `results/P3/candidates/lvef_lte45_global/`
- Progress: `reports/P3_lvef-qrs.progress.md`

## Problemi aperti
- statsmodels non disponibile: Wald raggruppato implementato a mano (numpy), non confrontato con una libreria.
- Interazioni testate solo con Ridge lineare; interazioni non lineari non esplorate (gli alberi locali del test (i) le coprono in parte).
- QRS>=120 non distingue blocco di branca sinistra da destra né da ritardo intraventricolare aspecifico (nessun dato).
- Bootstrap delle coppie non per paziente (IC un po' stretti); matching sul solo vicino più prossimo, 20% migliori.
- Nessuna validazione esterna; nessun dato su terapie/comorbidità. Il test set non è stato toccato.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- QRS_MODIFICA: inconclusiva (1/3 test pienamente concordi; non serve P3b)
- (i) stratificazione QRS val: ΔR² locale-globale +0.007 [0.001,0.013] (<100), +0.004 [-0.014,0.023] (100-119), +0.009 [-0.007,0.024] (>=120): nessun beneficio rilevante
- (ii) interazione train: Wald per paziente p<1e-4 (continuo) / 0.0069 (>=120), ma CV raggruppata ΔMSE +0.28 [-0.03,0.56] (R² 0.1400→0.1413): trascurabile
- (iii) coppie: |ΔQRS| grandi vs piccole +2..+8 ms, QRS più largo nella LVEF bassa nel 58-72% (IC sopra 50%) con 4 MATCH_ON; effetto ridotto a 2-4 ms appaiando PR/QTc; è associazione, non modifica d'effetto
- MATCH_ON_TESTATI: A [età,sesso,freq]; B [età,sesso]; C [età,sesso,freq,PR,QTc]; D [età,freq,PR,QTc] (direzione uguale, ampiezza dipendente)
- LVEF_REG: MAE=9.54 [9.31-9.79], R2=0.163 [0.136-0.191] (hist_gb, val n=4112; baseline MAE 11.19, R2 -0.02)
- LVEF_LTE45_AUROC: 0.764 [0.748-0.782] (hist_gb, val, 866 positivi); candidato results/P3/candidates/lvef_lte45_global; 17 configurazioni su val
RACCOMANDAZIONE:
- Gate P3: nessun P3b (non "supportata"); solo note. In P6 segnalare che il miglior modello è scelto sul validation e che l'ipotesi QRS non ha evidenza forte (SHD non è scompenso cardiaco; associazione non causalità).
