# Report R1-P3 · review-lvef-qrs
- Agente: lvef-qrs-analyst
- Data: 2026-11-06 15:45
- Stato: OK

## Obiettivo
Follow-up della revisione P6 su P3: (1) rifare il test (iii) sulle coppie di pazienti simili eliminando le coppie reciproche duplicate (A→B e B→A) e facendo il bootstrap per paziente; (2) correggere la descrizione del filtro «20% di coppie più vicine», che con MATCH_ON B non filtrava nulla; (3) correggere `candidate.json` di `lvef_lte45_global`. Promemoria: lo studio riguarda la cardiopatia strutturale (SHD), che NON coincide con lo scompenso cardiaco.

## Input letti
- Handoff P0 (integrità OK, split per paziente), P1, P3 (verdetto QRS_MODIFICA inconclusiva; coppie con 4 MATCH_ON).
- `reports/P3_lvef-qrs.md`, `experiments/P3/lvef_qrs.py`, `results/P3/b3_pairs.csv`, `results/P3/candidates/lvef_lte45_global/candidate.json`. Nota: `agentic/PLAYBOOK.md` non contiene una sezione «R1-P3» propria (ricerca per «R1» senza risultati); mi sono attenuto all'obiettivo assegnato dall'orchestratore.
- Dati solo con `lib.load_data` (test chiuso, mai toccato).

## Cosa è stato fatto
Script `experiments/R1-P3/pairs_dedup.py`, smoke test con `debug` (2000 righe, `results/R1-P3/debug/`) e poi run completo.
1. Stessa costruzione di P3: per ogni paziente il vicino più prossimo (altro paziente) sulle feature MATCH_ON standardizzate, sesso pesato x10; train = 1 ECG casuale per paziente (seed 42), replica su validation (4112 pazienti con LVEF).
2. **Deduplica**: le coppie sono trattate come non ordinate; se i→j e j→i sono entrambe presenti ne resta una (la coppia è tenuta una sola volta).
3. **Filtro 20%** applicato dopo la deduplica sul quantile 0.20 della distanza (`dist <= cutoff`), riportando cutoff, quota di distanze esattamente 0 e coppie tenute.
4. **Bootstrap per paziente (multiway)**: a ogni replica si ricampionano i pazienti con reinserimento (500 repliche); ogni coppia pesa w_i·w_j, quindi un paziente presente in più coppie è ricampionato come unità. Mediane pesate.
5. «Grande» = |ΔLVEF|≥15, «piccola» <5. Misure: differenza delle mediane di |ΔQRS| e frazione di coppie con QRS più largo nel membro con LVEF più bassa, con IC95%.
6. Descrizione del candidato riscritta con `lib.save_candidate` (stesso modello e codice, solo `candidate.json`).

## Risultati
**Filtro 20%: cosa succedeva davvero.** Nelle coppie non deduplicate di P3 (e anche dopo la deduplica) il taglio sul quantile 0.20 della distanza non seleziona il 20% quando molte distanze valgono 0 (età intera + sesso esatto = pareggi):

| MATCH_ON | split | coppie dirette | uniche (dopo dedup) | distanza = 0 | cutoff q20 | coppie tenute (quota) | pazienti in coppie / pool |
|---|---|---|---|---|---|---|---|
| A età, sesso, freq | train | 23637 | 19256 | 76.7% | 0 | 14776 (76.7%) | 20025 / 23637 |
| A | val | 4112 | 3209 | 28.5% | 0 | 913 (28.5%) | 1620 / 4112 |
| B età, sesso | train | 23637 | 23491 | 100% | 0 | 23491 (100%) | 23637 / 23637 |
| B | val | 4112 | 3971 | 99.9% | 0 | 3968 (99.9%) | 4111 / 4112 |
| C età, sesso, freq, PR, QTc | train | 23637 | 17112 | 0.2% | 0.145 | 3426 (20%) | 5858 / 23637 |
| C | val | 4112 | 3000 | 0.0% | 0.231 | 601 (20%) | 1029 / 4112 |
| D età, freq, PR, QTc | train | 23637 | 17087 | 0.3% | 0.120 | 3419 (20%) | 5819 / 23637 |
| D | val | 4112 | 2973 | 0.0% | 0.191 | 596 (20%) | 1021 / 4112 |

Correzione: il «filtro 20%» vale solo per C e D. Per **B** il filtro è inefficace: quasi tutte le distanze sono 0, quindi `n_pairs=23637` in P3 corrispondeva a tutti i pazienti del pool (una coppia per paziente, con reciproche duplicate); dopo la deduplica le coppie uniche sono 23491 (train). Per **A** nel train il filtro tiene il 76.7% (pareggi a distanza 0), non il 20%. Le coppie reciproche duplicate erano tra lo 0.6% (B train) e il 28% (C, D train) delle coppie dirette (A train 23637→19256, 18.5%).

**Test (iii) rifatto: coppie uniche, bootstrap per paziente.**

| MATCH_ON | split | coppie grandi/piccole | mediana |ΔQRS| grandi vs piccole (ms) | differenza [IC95%] | QRS più largo nella LVEF bassa: % [IC95%] | ΔQRS(LVEF bassa) mediano ms [IC] |
|---|---|---|---|---|---|---|
| A | train | 5219 / 3605 | 18 vs 12 | 6 [4, 8] | 66.3% [64.0-68.7] | 10 [8, 12] |
| A | val | 311 / 220 | 18 vs 11 | 7 [0, 14] | 68.8% [59.1-77.3] | 12 [6, 18] |
| B | train | 8896 / 5027 | 18 vs 12 | 6 [2, 10] | 61.8% [55.0-69.7] | 8 [4, 14] |
| B | val | 1423 / 880 | 20 vs 12 | 8 [2.95, 12] | 71.3% [65.2-77.0] | 14 [10, 20] |
| C | train | 1038 / 901 | 14 vs 10 | 4 [0, 6] | 58.2% [52.3-63.3] | 4 [2, 6] |
| C | val | 188 / 161 | 14 vs 10 | 4 [-2, 8] | 59.6% [47.4-71.3] | 6 [0, 12] |
| D | train | 1093 / 856 | 16 vs 12 | 4 [2, 6] | 61.1% [56.3-65.8] | 6 [4, 10] |
| D | val | 171 / 159 | 16 vs 14 | 2 [-4, 9.05] | 62.6% [49.9-74.4] | 10 [0, 16] |

Confronto con P3: la **direzione non cambia** (QRS più largo nel membro con LVEF più bassa in 58–71% delle coppie; nel train l'IC è sempre sopra 50%), ma gli IC sono più larghi, come atteso, perché la dipendenza tra coppie dello stesso paziente è ora considerata. Le stime puntuali di ampiezza sono 2–8 ms (A,B) e 2–4 ms (C,D). Nel train la differenza |ΔQRS| ha IC che esclude 0 per A, B, D e tocca 0 per C (0–6). In validation (n coppie molto minore) gli IC includono 0 per A, C, D; per B no. Con C e D in validation anche la frazione di QRS più largo ha IC che include 50% (C 47–71%, D 50–74%). Il QRS è misurato in passi di 2 ms, quindi le mediane sono quantizzate. Rho di Spearman tra |ΔLVEF| e |ΔQRS| 0.10–0.19 (associazione debole). Nota: i valori del vecchio P3 con n coppie di validation (es. A 1083, B 4109) erano diversi perché includevano reciproche duplicate e un filtro di quantile diverso nei pareggi.

**Effetto sul verdetto.** La conclusione di P3 non cambia: test (iii) concorde in direzione ma di ampiezza modesta, che si riduce appaiando PR/QTc e che in validation è statisticamente incerta; test (i) non concorde; test (ii) significativo come p ma trascurabile in CV. **QRS_MODIFICA resta «inconclusiva»** (≥2 test su 3 pienamente concordi non raggiunti). Cautela: coppie simili sulle sole feature tabulari restano pazienti clinicamente diversi (terapie, comorbidità, ischemia, dispositivi); è un'associazione marginale tra QRS e LVEF, non causalità né dimostrazione che il QRS modifichi la relazione tra le altre feature e la LVEF.

**Candidato `lvef_lte45_global`.** La descrizione precedente diceva «addestrato sul train (solo righe con LVEF)», ma il modello era addestrato con `lib.get_xy(tr_all, "lvef_lte_45_flag")`, cioè su **tutte le 72475 righe di train con il flag disponibile** (6457 senza `lvef_value` numerico), e valutato su tutte le 4626 righe di validation con flag (866 positivi). `candidate.json` ora dice questo; modello, `candidate.py` e metriche invariati (AUROC val 0.764 [0.748-0.782], AUPRC 0.443). Il modello è il migliore di 5 scelto sul validation: AUROC leggermente ottimistica. Non è stato verificato come il flag sia valorizzato nelle 6457 righe senza lvef_value; il flag è l'etichetta e non una feature, quindi nessun leakage in ingresso.

## Decisioni e motivazioni
- Deduplica prima del filtro quantile: così n coppie e quantile riflettono coppie distinte.
- Bootstrap multiway sui pazienti (non sulle coppie): è la misura di incertezza coerente con pazienti ripetuti in più coppie. Un bootstrap a due vie su coppie con vicini condivisi può essere ancora lievemente anti-conservativo; non è stato aggiunto altro.
- Non cambiati: soglie |ΔLVEF| 15/5, K=8, sesso x10; non si è provato un filtro di distanza assoluta (aggiungerebbe configurazioni su validation).
- Verdetto QRS_MODIFICA invariato (inconclusiva); non serve P3b.

## Configurazioni provate
4 configurazioni su validation (una per MATCH_ON, replica su val), registrate con `log-exp --n-configs 1` (E0016–E0019). Lo smoke test e la parte su train non usano il validation per scegliere. Totale P3 + R1-P3 su validation: 17 + 4 = 21 (le 4 coppie di R1-P3 ripetono le stesse analisi di P3 con metodo corretto: non sono scelte nuove).

## Artefatti
- Script: `experiments/R1-P3/pairs_dedup.py`
- Risultati: `results/R1-P3/pairs_dedup.csv` (aggregati), `results/R1-P3/candidate_fix.json`; smoke in `results/R1-P3/debug/`
- Candidato corretto (solo descrizione): `results/P3/candidates/lvef_lte45_global/candidate.json`
- Report originale: `reports/P3_lvef-qrs.md`

## Problemi aperti
- Il report `reports/P3_lvef-qrs.md` conserva i numeri vecchi del test (iii); i valori validi sono quelli di questo report.
- Il filtro 20% con A (train) e B è dominato da pareggi di distanza 0; una scelta più pulita (es. tolleranza esplicita su età) non è stata provata.
- QRS≥120 non distingue blocco di branca sinistra da altro; nessun dato su terapie/comorbidità. Test set non toccato.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- QRS_MODIFICA: inconclusiva (invariata dopo dedup + bootstrap per paziente)
- (iii) coppie uniche: |ΔQRS| grandi vs piccole +2..+8 ms; train A 6 [4,8], B 6 [2,10], C 4 [0,6], D 4 [2,6]; val IC includono 0 per A, C, D
- QRS più largo nella LVEF bassa: 58–66% train (IC sotto tutti >50%), 60–71% val (C 47–71%, D 50–74%)
- Filtro 20% corretto: B tiene 23491 coppie (100%, distanze=0) e A train 76.7%; il filtro funziona solo per C e D (20%)
- Coppie reciproche rimosse: A train 23637→19256, B 23637→23491, C 17112, D 17087
- MATCH_ON_TESTATI: A, B, C, D (direzione uguale, ampiezza dipendente; associazione non causalità)
- CANDIDATO lvef_lte45_global: addestrato su 72475 righe train con flag (6457 senza lvef_value); AUROC val 0.764 [0.748-0.782] invariata
- Configurazioni su val: +4 (E0016–E0019)
RACCOMANDAZIONE:
- P6: usare i numeri di R1-P3 per il test (iii); verdetto inconclusiva confermato. Candidato lvef_lte45_global con descrizione corretta, AUROC ottimistica (scelto su val), comparatore non pertinente.
