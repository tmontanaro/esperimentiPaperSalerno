# Report P6 · review
- Agente: skeptic-reviewer
- Data: 2026-11-06 16:10
- Stato: OK

## Obiettivo
Secondo round della revisione indipendente prima dell'apertura (unica) del test set. Verificare che le tre revisioni chieste al round 1 (R1-P4, R1-P5, R1-P3) siano state fatte davvero, rifare i controlli che al round 1 non era stato possibile eseguire (esecuzione di `candidate.predict`, ricalcolo dei numeri) e decidere quali candidati possono andare al test. Promemoria: SHD (cardiopatia strutturale moderata o più) NON coincide con lo scompenso cardiaco.

## Input letti
- Report del round 1: `reports/P6_review.md`, copiato senza modifiche in `reports/P6_review.round1.md`.
- Handoff da `python3 tools/state.py brief`: P0-P5, R1-P3, R1-P4, R1-P5; report completi `reports/R1-P3_review-lvef-qrs.md`, `reports/R1-P4_review-waveform.md`, `reports/R1-P5_review-clusters.md`.
- Script: `experiments/R1-P4/strat.py`, `experiments/R1-P3/pairs_dedup.py`; ricerca testuale su tutto `experiments/` di `allow_test`, `ALLOW_TEST`, `split == test`, `ECHO_ONLY`, `lvef_value`.
- File aggregati: `results/registry.csv`, `results/P5/local_vs_global.csv`, `results/R1-P3/pairs_dedup.csv`, `results/R1-P4/val_stratified.csv`, `results/R1-P4/candidate_check.json`, `results/P4/val_pred_emb.npz` (predizioni, usate solo per calcolare aggregati).
- Candidati: `results/P4/candidates/cnn_multilabel`, `results/P3/candidates/lvef_lte45_global`, `results/P1/candidates/global_best` (comparatore).

## Cosa è stato fatto
Al round 1 l'esecuzione di comandi era negata; in questo round è stato possibile eseguire codice.
1. Script `experiments/P6/verify_round2.py`: smoke test con `debug` (2000 righe, output in `results/P6/debug/`), poi esecuzione completa sul validation. Calcola: configurazioni per fase dal registro; conteggio dei confronti appaiati di P5; ristratificazione indipendente CNN vs hist_gb (bootstrap con seed 7, diverso da quello di R1-P4, 500 ricampionamenti); contratto dei 3 candidati su 300 righe di validation (seed 123, diverso dalle 500 righe di R1-P4); AUROC del candidato LVEF≤45 su tutto il validation; coerenza tra `lvef_lte_45_flag` e `lvef_value`.
2. Dal punto 1 è emersa un'anomalia nelle etichette. Due controlli in linea, salvati in `results/P6/lvef_label_check.json` e `results/P6/missing_lvef_check.json`, ne misurano l'effetto: prevalenza di tutti i 12 flag nelle righe senza `lvef_value`, AUROC dei modelli con e senza quelle righe.
3. Confronto riga per riga degli Handoff R1 con i file `results/R1-*` (tolleranza 0.01 AUROC).

## Risultati
**A. Le revisioni del round 1 sono state fatte (verificate dai file, non dalle narrazioni).**

| Revisione richiesta | Esito | Evidenza |
|---|---|---|
| P4: equità della CNN per etnia, setting, sesso ed età, con IC appaiato contro hist_gb | FATTA | `results/R1-P4/val_stratified.csv`: 16 strati. Il mio ricalcolo indipendente (`results/P6/strat_recalc.csv`) riproduce tutti gli AUROC della CNN entro 0.003 |
| P4: `candidate.predict` su 500 righe di validation | FATTA e ripetuta | R1-P4: max\|Δ\| 1.96e-4. Il mio controllo su 300 righe diverse: colonne `score`,`in_scope`; indice identico; in_scope 100%; 0 NaN; max\|score − predizione salvata\| = 1.1e-4 |
| P5: registrare le 43 configurazioni mancanti | FATTA | registro: P5 125 (di cui 47 con val), R1-P5 43 (E0021), cioè 168 |
| P5: dichiarare i 2 IC appaiati positivi su 40 e il p=0.77 | FATTA | ricalcolo da `local_vs_global.csv`: 40 confronti, 2 con IC>0, 1 con IC<0 |
| P5: ARI dominato dal sesso | FATTA, ma solo come interpretazione | nessuna analisi di sensibilità senza il sesso (lo ammettono) |
| P3: coppie deduplicate, bootstrap per paziente, descrizione corretta del filtro 20% | FATTA | `pairs_dedup.py` r.35-36 (deduplica), r.57-58 (bootstrap multiway per paziente); i numeri del CSV coincidono con l'Handoff (es. A train 6 [4, 8], C val 4 [-2, 8], B frac. distanze nulle 1.000) |
| P3: descrizione di `candidate.json` | FATTA | il json riporta "n=72475 ECG, di cui 6457 senza lvef_value"; il modello non è cambiato (AUROC val ricalcolato 0.7643) |

**B. Leakage: nessun leakage nei dati di ingresso.** Gli script R1 non toccano il test: l'unico `allow_test` in `experiments/` è `P4/train.py` r.162, condizionato a `ECHONEXT_ALLOW_TEST=1`, che imposta solo `final_eval.py`. Nessuna colonna di `ECHO_ONLY_COLS` entra nei modelli: `lvef_value` compare solo come esito e come variabile di appaiamento in P3/R1-P3. Nessun modello è addestrato su righe di validation.

**C. NUOVO RILIEVO: le righe senza `lvef_value` hanno tutti i flag a 0 per costruzione.** È un artefatto delle etichette, non un leakage di ingresso.

| | train | validation |
|---|---|---|
| righe senza `lvef_value` | 6457 (8.9%) | 514 (11.1%) |
| prevalenza di OGNUNO dei 12 flag (SHD incluso) in quelle righe | 0.000 | 0.000 |
| età media senza / con LVEF | 57.6 / 61.6 | 57.2 / 63.3 |
| `lvef_lte_45_flag` coerente con `lvef_value ≤ 45` dove c'è il valore | 100% | 100% |

In queste righe nessuna misura è stata registrata e sono tutte "negative". Sono più giovani e più spesso ambulatoriali, quindi riconoscibili in parte anche dall'ECG: la CNN dà loro un punteggio medio di 0.222, contro 0.308 dei veri negativi con LVEF misurata. Effetto sulle metriche (validation):

| Modello | Target | AUROC su tutte le righe [IC95%] | AUROC sulle sole righe con LVEF misurata [IC95%] | Differenza |
|---|---|---|---|---|
| CNN `cnn_multilabel` | SHD | 0.823 [0.811-0.834], n=4626, 1990 pos | 0.812 [0.798-0.824], n=4112, 1990 pos | −0.011 |
| hist_gb `global_best` | SHD | 0.746 [0.733-0.761] | 0.732 [0.716-0.746] | −0.015 |
| hist_gb `lvef_lte45_global` | LVEF≤45 | 0.764 [0.748-0.782], 866 pos | 0.749 [0.731-0.769], n=4112, 866 pos | −0.015 |
| testa LVEF≤45 della CNN (non candidata) | LVEF≤45 | 0.889 | 0.881 | −0.008 |

Le conclusioni restano: la CNN batte il tabulare anche sulle righe con LVEF misurata (0.812 contro 0.732) e il candidato LVEF resta "moderato". Tutte le AUROC riportate in P1-P5 sono però gonfiate di circa 0.01-0.015 da un gruppo di negativi "facili" la cui etichetta è dubbia: potrebbero essere ecocardiogrammi incompleti, non cuori sani. Nessuna fase precedente l'aveva notato. P0 riportava solo l'8.9% di LVEF mancanti; R1-P3 scrive "non è stato verificato come il flag sia valorizzato".

**D. Riproduzione dei numeri R1 (tolleranza 0.01 AUROC): 0 discrepanze.** R1-P4: AUROC CNN per strato identici (asian 0.788, procedurale 0.748, ≥75 0.788, 65-74 0.802). R1-P3: le 8 righe di `pairs_dedup.csv` coincidono con le tabelle del report. R1-P5: 40/2/1 confermato. Unica fragilità: l'IC appaiato "asian" di R1-P4 vale +0.007..+0.170 con il loro seed e −0.001..+0.185 con il mio (n=134, 62 positivi). La frase "IC>0 in 15/16 strati" dipende quindi dal seed del bootstrap: con il mio sono 14/16. La CNN resta sopra hist_gb in 16/16 come stima puntuale.

**E. Multiplicità e uso del validation.** Registro attuale: P1 10, P2 308, P3 17, P4 2, P5 125 (47 con val), R1-P3 4, R1-P4 1, R1-P5 43, P6 1 (E0022, questa verifica descrittiva). Configurazioni che hanno usato il validation: 433. Restano fuori dal validation 1950 regole P2 e 78 configurazioni di clustering P5, valutate solo sul train. Sottogruppi e cluster sono stati definiti sul solo train e verificati in validation (procedura corretta). Il p empirico di P5 include la selezione delle feature locali e il massimo su cluster e modelli. Gli IC del validation sono per paziente (1 ECG per paziente); le coppie R1-P3 usano ora un bootstrap per paziente.

**F. Definizioni.** Nessuna etichetta "credibile" assegnata (0 regole P2, 0 cluster P5), e i controlli lo confermano: stenosi aortica cluster 2 ha AUROC locale 0.812 [0.761-0.868], sotto il globale 0.844, e p=0.77. `QRS_MODIFICA` = "inconclusiva": i tre test non concordano. (i) Nessun beneficio dalla stratificazione (ΔR² ≤ 0.009). (ii) Interazione significativa come p ma con R² in CV da 0.1401 a 0.1413. (iii) Associazione marginale di 2-8 ms, con IC che in validation include 0 per A, C e D. L'etichetta è corretta, anzi generosa.

**G. Affermazioni (Handoff R1, 19 righe).**

| Fase | Riga | Giudizio | Motivo |
|---|---|---|---|
| R1-P4 | CNN > hist_gb 16/16, IC>0 15/16 | DEBOLE | 16/16 regge; 15/16 dipende dal seed (asian −0.001 con seed 7) |
| R1-P4 | AUROC più bassa/alta per strato | SUPPORTATA | riprodotta |
| R1-P4 | vantaggio da +0.054 a +0.120 | SUPPORTATA | riprodotto (IC simili) |
| R1-P4 | sesso 0.817 / 0.822 | SUPPORTATA | riprodotto |
| R1-P4 | candidate.predict | SUPPORTATA | ripetuto su 300 righe diverse, max\|Δ\| 1.1e-4 |
| R1-P4 | configurazioni +1 | SUPPORTATA | E0020 |
| R1-P5 | N_CONFIGURAZIONI 168 | SUPPORTATA | 125 + 43 nel registro |
| R1-P5 | 40 confronti, 2 IC>0, 1 IC<0 | SUPPORTATA | ricalcolato |
| R1-P5 | P_EMPIRICO 0.77/0.52/0.94 | SUPPORTATA | permutation.csv (round 1) |
| R1-P5 | CLUSTER_CREDIBILI [] | SUPPORTATA | 0.761 < 0.844 |
| R1-P5 | ARI dominato dal sesso | DEBOLE | plausibile ma non testato (nessun ARI senza sesso) |
| R1-P3 | QRS_MODIFICA inconclusiva | SUPPORTATA | vedi F |
| R1-P3 | differenze \|ΔQRS\| | SUPPORTATA | pairs_dedup.csv |
| R1-P3 | frazione QRS più largo | SUPPORTATA | train 58.2-66.3%, IC inferiori ≥ 0.523 |
| R1-P3 | filtro 20% | SUPPORTATA | frac_tenute 0.767/1.000/0.200/0.200 |
| R1-P3 | coppie reciproche rimosse | SUPPORTATA | n_dirette → n_uniche |
| R1-P3 | MATCH_ON A-D | SUPPORTATA | 4 insiemi nel CSV |
| R1-P3 | candidato lvef_lte45: 72475 righe, AUROC 0.764 | DEBOLE | numeri esatti, ma le 6457 righe senza LVEF sono tutte negative per costruzione; sulle righe con LVEF l'AUROC è 0.749 |
| R1-P3 | configurazioni +4 | SUPPORTATA | E0016-E0019 |

Round 2: 16 supportate, 3 deboli, 0 non supportate. Aggiornamento delle 27 righe del round 1: "P3 coppie" e "P5 N_CONFIGURAZIONI" diventano supportate grazie alle revisioni. "P3 LVEF_LTE45_AUROC" scende a debole per il rilievo C. "P5 ARI" resta debole. Round 1 aggiornato: 25/2/0. **Totale: 41 supportate / 5 deboli / 0 non supportate (46 righe).**

**H. Senso clinico ed equità.**
- SHD ≠ scompenso. L'etichetta è composita: AS tabulare 0.844 contro versamento pericardico 0.678. Centro unico. Solo pazienti con un ecocardiogramma (bias di selezione: prevalenza 43-52%, lontana da uno screening).
- Equità della CNN (R1-P4, verificata): nessun gruppo in cui la CNN fallisce. La CNN cala con l'età (<50: 0.831; ≥75: 0.788 [0.761-0.813]) ed è più bassa nel setting procedurale (0.748 [0.674-0.818], n=177). Gli IC si sovrappongono e non sono confronti formali tra gruppi.
- Calibrazione grezza sul validation (prevalenza 0.430): punteggio medio CNN 0.455, hist_gb 0.503 (risente dello spostamento di prevalenza tra train, 0.524, e validation). La calibrazione non è stata valutata formalmente da nessuna fase.

**I. Contratto dei candidati.** Tutti e tre i `candidate.py` sono stati eseguiti su 300 righe di validation. Restituiscono `score` e `in_scope` con indice identico al DataFrame in ingresso, 0 NaN, in_scope 100%. Ricalcolata su tutto il validation, l'AUROC di `lvef_lte45_global` coincide con quella dichiarata (0.7643).

## Decisioni e motivazioni
- **VERDETTO PASS.** Le revisioni richieste al round 1 sono fatte e riprodotte, non c'è leakage di ingresso e i contratti dei candidati funzionano. Il rilievo C non è un leakage: è una proprietà delle etichette del dataset, che vale identica in train, validation e (presumibilmente) test. Non cambia nessuna conclusione: la CNN supera il tabulare anche sulle righe con LVEF misurata (0.812 contro 0.732) e nessun candidato va cambiato. L'ho quantificato qui, quindi non serve un'altra fase. Lo rendo un'analisi **pre-specificata prima dell'apertura del test** (nota in DECISIONS.md): P7/P8 devono riportare anche la metrica sulle righe con `lvef_value` non nullo.
- **Candidati ammessi:**
  - `results/P4/candidates/cnn_multilabel`: una configurazione, epoca fissata a priori, stima di validation poco ottimistica. Validation 0.823 [0.811-0.834]; 0.812 [0.798-0.824] sulle righe con LVEF.
  - `results/P3/candidates/lvef_lte45_global`: scelto fra 5 modelli sul validation, ottimismo piccolo; accuratezza solo moderata, 0.749 [0.731-0.769] sulle righe con LVEF.
- **Nessun sottogruppo né cluster ammesso**: nessuno soddisfa i criteri del PLAYBOOK.
- Non ho modificato report, dati o candidati di altre fasi; non ho toccato il test.

## Configurazioni provate
1 valutazione descrittiva sul validation (E0022, `--uses-val --n-configs 1`): ristratificazione, contratto dei candidati, analisi di sensibilità sulle righe senza LVEF. Nessun modello è stato addestrato né scelto: questa verifica non introduce ottimismo da selezione.

## Artefatti
- `reports/P6_review.md` (questo file); `reports/P6_review.round1.md` (round 1, invariato).
- `experiments/P6/verify_round2.py`
- `results/P6/verify_round2.json`, `results/P6/strat_recalc.csv`, `results/P6/lvef_label_check.json`, `results/P6/missing_lvef_check.json`; smoke test in `results/P6/debug/`.

## Problemi aperti
- **Etichette con LVEF mancante** (rilievo C): non sappiamo se le 514 righe di validation (e le 6457 di train) siano cuori sani o ecocardiogrammi incompleti codificati come 0. Va chiesto ai partner clinici o al dizionario dati di EchoNext. Tutte le AUROC del progetto andrebbero lette anche sul sottoinsieme con LVEF misurata (circa −0.01/−0.015).
- `tools/final_eval.py` (che non posso modificare) probabilmente calcola le metriche solo su tutte le righe di test. Se P7 non può ottenere anche il sottoinsieme con LVEF misurata, P8 deve dichiarare l'effetto stimato qui sul validation.
- Comparatore unico in FROZEN (modello del composito): per `lvef_lte45_global` il confronto con il comparatore non ha senso.
- La testa LVEF≤45 della CNN (0.881 sulle righe con LVEF) è molto migliore del candidato tabulare ma non è un candidato: promuoverla ora sarebbe una scelta fatta guardando il validation; decisione lasciata all'umano.
- Un solo seed della CNN; calibrazione non valutata; ARI dei cluster senza il sesso non testato; centro unico.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- VERDETTO: PASS (revisioni R1-P4, R1-P5, R1-P3 fatte e riprodotte; nessun leakage di ingresso; 0 discrepanze numeriche oltre 0.01 AUROC)
- CANDIDATI_AMMESSI: [results/P4/candidates/cnn_multilabel, results/P3/candidates/lvef_lte45_global]; 0 sottogruppi, 0 cluster
- AFFERMAZIONI: 41/5/0 (round 2: 16/3/0; deboli: R1-P4 "IC>0 15/16" dipende dal seed, asian −0.001 con seed 7; ARI senza sesso non testato; AUROC LVEF≤45 gonfiata)
- NUOVO: righe senza lvef_value (train 6457, val 514) hanno tutti i 12 flag = 0; AUROC val sulle sole righe con LVEF: CNN 0.812 [0.798-0.824], hist_gb 0.732 [0.716-0.746], LVEF≤45 0.749 [0.731-0.769] (−0.011/−0.015)
- candidate.predict (300 righe val, 3 candidati): score+in_scope, indice identico, 0 NaN; CNN max|Δ| vs salvato 1.1e-4
- QRS_MODIFICA: inconclusiva confermata ((i) ΔR² ≤ 0.009; (ii) R² CV 0.1401→0.1413; (iii) 2-8 ms, IC val con 0 per A, C, D)
- Equità CNN: 16/16 strati sopra hist_gb; minimi procedurale 0.748 [0.674-0.818] e ≥75 anni 0.788 [0.761-0.813]
- Multiplicità: 433 configurazioni sul validation (E0001-E0022) + 1950 regole e 78 clustering solo sul train
RACCOMANDAZIONE:
- Orchestratore/umano: approvare P7 con i 2 candidati; congelare in FROZEN.json.
- P7: oltre all'AUROC su tutto il test, riportare (pre-specificato qui) quella sulle righe con lvef_value non nullo; se final_eval.py non lo consente, dichiararlo. Ignorare il comparatore per lvef_lte45_global.
- P8: SHD ≠ scompenso; righe senza LVEF tutte negative per costruzione; centro unico, solo pazienti con eco; calibrazione non valutata.
