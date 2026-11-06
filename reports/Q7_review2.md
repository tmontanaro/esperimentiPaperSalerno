# Report Q7 · review2
- Agente: skeptic-reviewer2
- Data: 2026-11-06 21:30
- Stato: OK

## Obiettivo
Revisione indipendente e scettica della pipeline 2 (Q0-Q6) prima della seconda lettura del test (Q8). Bisognava controllare con codice
scritto apposta (non con gli script delle altre fasi): separazione dei pazienti tra fold, riproducibilità dei numeri, popolazione «clean»,
test mai toccato, confronti multipli, affermazioni non supportate, integrità dei 5 candidati congelati. Promemoria: SHD (cardiopatia
strutturale moderata o peggiore) NON coincide con lo scompenso cardiaco.

## Input letti
- `python3 tools/state2.py brief` e Handoff di Q0-Q6; `agentic/PLAYBOOK2.md` (§Regole speciali, §Q7); `agentic/FROZEN2.json`; `results/Q6/preregistration.md`.
- Artefatti aggregati: `results/Q1/main/*` (OOF, metriche, config), `results/Q1/all_rows`, `results/Q2/*`, `results/Q3/q3_results.json`,
  `results/Q4/{strata,error_rules,clusters}.csv`, `results/Q4/cluster_summary.json`, `results/Q5/calib_ops.json`, `results/registry.csv`.
- Codice letto (non eseguito per riprodurre): `tools/cnn_cv.py` (come si costruiscono fold, `n_train`, candidati), `tools/final_eval2.py`, `lib/pipe2_lib.py`.

## Cosa è stato fatto
Tutti gli script sono in `experiments/Q7/` e ricalcolano da soli le fold (`zlib.crc32("42-<patient_key>") % 5`, senza `fold_of`),
le AUROC con `sklearn.metrics.roc_auc_score` e il bootstrap per paziente (codice proprio). Ognuno è stato provato prima in piccolo
(`debug_n=2000` o 300 righe), poi sull'intero pool di sviluppo (train + validation, 77 101 ECG, 30 844 pazienti; mai il test).
1. `check_folds_auc.py`: per tutte le 10 run CNN in `results/Q*/` (56 file `oof_seed*_fold*.npz`) verifica che le righe predette siano
   esattamente il fold, che nessun paziente predetto stia nei dati di addestramento attesi e che `n_train` coincida con
   «altre fold ∩ LVEF noto» (clean) o «altre fold» (all). Riproduce 216 AUROC (tutte le run × 12 etichette × clean/all) contro
   `metrics_clean.csv`/`metrics_all.csv`; IC95% per paziente propri per 5 etichette; MAE/R² della LVEF; differenza Q2 3 derivazioni.
2. `check_test_frozen.py`: test chiuso (cartella `results/final2/`, ricerca di `allow_test`/«test» negli script `experiments/Q*/`),
   30 impronte sha256 di `FROZEN2.json`, coerenza `candidate.py`/`candidate.json`/pesi/target, ordine temporale preregistrazione → pesi → congelamento.
3. `check_contract.py`: carica ogni candidato e chiama `predict(df, lib)` su 1 500 righe di dev; confronta con la mia ricostruzione
   (pesi caricati a mano, forme d'onda lette direttamente) e con le OOF di Q1 sugli stessi ECG (per accorgersi di un'uscita sbagliata).
4. `check_q3_q5.py`: Holm ricalcolata (Q3: 3 famiglie; Q4: 90 strati e 32 regole), regressione logistica di Q3(b) rifatta da zero,
   due strati di Q4, ECE/Brier e punto operativo «sensibilità 90%» di Q5, sensibilità alle righe senza LVEF (Q1), set `eight` (Q2).
5. `check_occlusion.py`: occlusione di ogni canale (azzeramento) sul modello seed 0 / fold 0, 4 000 ECG clean del fold 0, IC appaiato per paziente.

## Risultati
Tutti i valori sono in CV fuori-fold sul pool di sviluppo (train + validation), popolazione clean salvo indicazione.

**1-3. Fold, riproduzione, popolazioni**

| Controllo | Esito |
|---|---|
| Pazienti in più di un fold | 0 su 30 844 |
| File OOF controllati | 56 (10 run); righe predette = fold intero in 56/56 |
| Pazienti in comune addestramento/verifica | 0 (massimo su tutti i file) |
| `n_train` atteso (clean esclude righe senza LVEF) | corretto in 56/56 |
| Righe senza LVEF con almeno un'etichetta positiva | 0 su 6 971 (conferma il pattern anomalo dichiarato) |
| AUROC riprodotte con sklearn | 216; differenza massima 0.0000 (tolleranza 0.005); n e n positivi identici |

IC95% ricalcolati con il mio bootstrap per paziente (300 ricampionamenti), ensemble 3 seed, clean (n = 70 130):

| Etichetta | n pos | CNN AUROC [IC95%] | Tabulare AUROC [IC95%] | Dichiarato |
|---|---|---|---|---|
| SHD | 39 948 | 0.815 [0.809-0.821] | 0.712 [0.704-0.720] | 0.815 [0.809-0.822] / 0.712 |
| LVEF ≤ 45% | 17 828 | 0.869 [0.861-0.876] | 0.720 [0.710-0.728] | 0.869 |
| Disfunzione VD | 9 965 | 0.844 [0.832-0.854] | 0.718 [0.702-0.730] | 0.844 |
| Stenosi aortica | 3 171 | 0.811 [0.795-0.824] | 0.841 [0.827-0.855] | 0.811 (tab 0.841) |
| Insufficienza aortica | 940 | 0.692 [0.653-0.726] | 0.602 [0.561-0.646] | 0.692 [0.652-0.732] |

Altri numeri riprodotti: AUROC SHD per seed 0.8075 / 0.8085 / 0.8079 (DS < 0.001); LVEF continua CNN MAE 7.48, R² 0.482 contro
tabulare 10.45 / 0.156; Q2 3 derivazioni contro tutte (seed 0): SHD −0.0056, LVEF ≤ 45% −0.0082; `eight` +0.0021 / +0.0004;
sensibilità righe senza LVEF: +0.0020 (SHD), +0.0026 (LVEF ≤ 45%). Tutti coincidono con i report.

**4. Test chiuso**: `results/final2/` non esiste; 0 occorrenze di `allow_test` o della stringa «test» negli script `experiments/Q0-Q6`;
`results/final/` contiene solo `test_results.json` (data 2026-11-06 15:49, prima dell'inizio della pipeline 2 alle 17:36: viene da P7).

**5. Multiplicità**: registro `results/registry.csv` = 40 esperimenti, 725 configurazioni (pipeline 1: 509; pipeline 2: Q1 2, Q2 7, Q3 8,
Q4 137, Q5 58, Q6 2 = 214). Q4: 137 = 15 cluster + 90 strati + 32 regole, coincide con `counts.json`. Holm ricalcolata: Q4 strati
differenza massima 0.000 (19 strati significativi su 90, come dichiarato), Q4 regole 0.000 (11 su 32), Q3 famiglie a/b/c identiche
(a 0.006 ×3; b 0.792/0.576; c 0.464/0.488). Q3(b) rifatta da zero: ΔAUROC +0.00006 (dichiarato +0.0001 [−0.0005, +0.0006]).

**7. Candidati**

| Controllo | Esito |
|---|---|
| Numero | 5 (massimo 6) |
| Impronte sha256 (candidati + pesi + config) | 30/30 coincidono |
| Ordine temporale | preregistrazione 20:33:30 → pesi 20:35-20:44 → candidati 20:44 → congelamento 20:45:03 |
| Config final-fit = CONFIG_FINALE Q1 | sì (12 epoche, lr 0.002, reg 0.5, batch 128, clean, 3 seed); C5 canali [0, 1, 6] = set Q2 |
| `predict(df, lib)` | forma, indice, punteggi in [0, 1], 0 NaN per tutti e 5; differenza dalla mia ricostruzione 0.000 |
| Uscita giusta | correlazione con le OOF della stessa etichetta 0.90-0.94, contro ≤ 0.73 con le altre etichette |
| AUROC su righe di addestramento (solo sanity, ottimistica) | 0.87-0.94, contro OOF sulle stesse righe 0.81-0.86 |
| Regole Q6 | C1, C2 sempre; C3 VD, C4 stenosi = TARGETS_GOOD con ≥ 100 positivi, in ordine di AUROC; C5 = set minimo Q2. Rispettate |

**Occlusione (verifica di Q5)**, seed 0 fold 0, 4 000 ECG (2 296 SHD positivi), AUROC base 0.813: canale 6 −0.021 [−0.028, −0.012]
(Q5: −0.026 su 5 000 righe) è il calo più grande; canale 1 −0.011 [−0.017, −0.007], canale 0 −0.008, gli altri tra −0.000 e −0.006.

**6. Affermazioni di Q1-Q5**

| Fase | Affermazione | Giudizio | Motivo |
|---|---|---|---|
| Q1 | CNN SHD 0.815 [0.809-0.822], migliore del tabulare di +0.103 | supportata | riprodotta con sklearn e bootstrap proprio |
| Q1 | Variabilità tra seed < 0.001 | supportata | 0.8075-0.8085 |
| Q1 | TARGETS_GOOD: SHD, LVEF ≤ 45%, VD, stenosi aortica | supportata | IC inferiori 0.809 / 0.861 / 0.832 / 0.795 ≥ 0.75 |
| Q1 | Insufficienza aortica debole (0.692) | supportata | IC 0.653-0.726 |
| Q1 | Stenosi aortica: il tabulare è migliore | supportata | 0.841 [0.827-0.855] contro 0.811 [0.795-0.824] |
| Q1 | LVEF continua: CNN R² 0.482 contro 0.156 | supportata | riprodotta |
| Q1 | Le righe senza LVEF nel training cambiano poco (+0.002) | supportata | riprodotta; IC dichiarato esclude 0 ma effetto trascurabile |
| Q2 | 3 derivazioni «sufficienti» (−0.006 SHD, −0.008 LVEF) | debole | riprodotta, ma un solo seed e regola «≥ −0.01» puntuale: l'IC SHD dichiarato [−0.009, −0.003] esclude 0, cioè c'è una perdita reale anche se piccola |
| Q2 | `eight` senza perdita | supportata | +0.002 / +0.000 |
| Q3 | QRS non aggiunge nulla sopra la CNN (non_supportata) | supportata, con limite | ΔAUROC +0.00006 riprodotta; ma la CNN vede già il QRS: non dice nulla sull'ipotesi clinica riferita alle 7 misure tabulari |
| Q3 | CNN meglio del tabulare in ogni fascia di QRS | supportata | p Holm 0.006 in tutte e 3 le fasce |
| Q3 | Coppie discordanti: +1.5 ms di QRS | supportata | significativa ma clinicamente trascurabile, come dichiarato |
| Q4 | Nessun cluster credibile sugli embedding | supportata | 0/15; AUROC locale sempre sotto la globale (−0.05/−0.18) |
| Q4 | La CNN «peggiora» con QRS < 100, PR corto, QTc basso | debole | numeri riprodotti (SHD QRS<100 0.787), ma l'AUROC dentro uno strato dipende anche dalla composizione dei pazienti (strati più omogenei → AUROC più bassa); nello stesso strato il tabulare scende a 0.679, quindi la CNN mantiene il vantaggio; non è un difetto specifico della rete |
| Q4 | Regole sugli errori (11/32 reggono) | debole | l'errore assoluto è più alto dove la prevalenza è vicina a 0.5: in parte effetto di prevalenza (dichiarato per LVEF, vale anche per SHD) |
| Q5 | ECE SHD 0.031 grezza → 0.003 isotonica | supportata, con riserva | ECE grezza riprodotta (0.0307); ma l'IC dell'ECE isotonica [0.003-0.009] non contiene bene il valore puntuale: il bootstrap dell'ECE è distorto verso l'alto, quindi l'IC va letto con cautela |
| Q5 | Punto operativo sens 90% SHD: spec 0.446, PPV 0.683 | supportata | riprodotto: spec 0.446, PPV 0.683, soglia 0.296 |
| Q5 | Shift di prevalenza train 0.575 / validation 0.484 | supportata | riprodotto |
| Q5 | Canale 6 il più importante, «tutte le altre < 0.01» | debole | canale 6 primo riprodotto (−0.021); sul mio campione il canale 1 cala −0.011 [−0.017, −0.007], appena oltre 0.01 |

Conteggio: 14 supportate (di cui 3 con limite/riserva), 5 deboli, 0 non supportate.

## Decisioni e motivazioni
- **VERDETTO: PASS.** Nessuna discrepanza numerica oltre la tolleranza (massimo 0.000 sulle AUROC, 0.005 sull'occlusione che usa un
  campione diverso), nessuna fuga tra fold, test chiuso, impronte integre, regole di Q6 rispettate e scritte prima dell'addestramento.
- **Nessun candidato rimosso** da `agentic/FROZEN2.json` (file non modificato). Tutti e 5 discendono da regole fissate prima e hanno
  IC inferiore in CV ≥ 0.79.
- C4 (stenosi aortica) resta: la regola lo impone (TARGETS_GOOD) anche se il tabulare è migliore in CV; va letto come test della CNN, non come modello consigliato.
- Non ho registrato configurazioni con `log-exp`: le mie analisi sono riproduzioni, non hanno scelto nulla e non devono gonfiare il conteggio di multiplicità.

## Configurazioni provate
0 configurazioni nuove scelte (nessun `log-exp`). Le riproduzioni usano il pool di sviluppo (che contiene il validation) ma non selezionano
modelli, soglie o candidati. Totale nel registro invariato: 725 configurazioni, 40 esperimenti.

## Artefatti
- Script: `experiments/Q7/check_folds_auc.py`, `check_test_frozen.py`, `check_contract.py`, `check_q3_q5.py`, `check_occlusion.py`.
- Risultati (solo aggregati): `results/Q7/check_folds_auc.json`, `fold_checks.csv`, `auc_reproduction.csv`, `check_test_frozen.json`,
  `check_contract.json`, `check_q3_q5.json`, `check_occlusion.json`.

## Problemi aperti
- **Popolazione del criterio di conferma non scritta**: la preregistrazione dice «AUROC test» ma non se su tutte le righe o su quelle con LVEF noto.
  `final_eval2.py` calcola entrambe. La regola 4 di PLAYBOOK2 (decisa prima: risultato principale = clean) risolve: Q8 usi `test_lvef_known` per il criterio e riporti l'altra come sensibilità.
- **C5, riferimento CV favorevole**: l'AUROC CV di C5 (0.802) è di un solo seed, mentre il candidato è un ensemble di 3 seed (per tutte le derivazioni
  l'ensemble aggiunge circa +0.007: 0.808 → 0.815). Il criterio «calo ≤ 0.05» è quindi un po' più facile per C5.
- **Seconda lettura non del tutto indipendente**: chi ha progettato la pipeline 2 conosceva il risultato di P7 (CNN SHD 0.830 al test). Le scelte di Q1-Q6
  non sono state fatte sul test, ma la direzione «puntare sulla CNN» è informata dalla prima lettura. Va dichiarato in Q8/Q9.
- Multiplicità complessiva alta (725 configurazioni): i risultati dei sottogruppi e degli strati sono esplorativi; i candidati no (regole fisse).
- Non verificato: l'occlusione delle finestre temporali di Q5 e i cluster di Q4 non sono stati rifatti (solo Holm e numeri aggregati ricontrollati).
- Limiti già dichiarati che restano: un solo centro, solo pazienti con eco, righe senza LVEF anomale (motivo non chiaribile), nomi delle derivazioni toraciche non verificabili.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- VERDETTO: PASS
- CANDIDATI_AMMESSI: [C1_shd_all, C2_lvef45_all, C3_rv_all, C4_as_all, C5_shd_I_II_V1] (5/5, nessuno rimosso; impronte sha256 30/30 integre)
- AFFERMAZIONI: 14/5/0 (supportate/deboli/non supportate) su Q1-Q5
- DISCREPANZE_NUMERICHE: nessuna; 216 AUROC riprodotte con sklearn, differenza massima 0.000; occlusione canale 6 −0.021 [−0.028, −0.012] contro −0.026 dichiarato (campione diverso)
- FOLD: 0 pazienti condivisi tra addestramento e verifica in 56 file OOF (10 run); n_train clean corretto 56/56
- TEST_CHIUSO: results/final2 assente; 0 usi di allow_test in experiments/Q*; results/final solo da P7
- MULTIPLICITA: 725 configurazioni nel registro (214 nella pipeline 2); Holm di Q3 e Q4 ricalcolata identica
- RISCHI_RESIDUI: popolazione del criterio di conferma non scritta (usare clean, regola 4); riferimento CV di C5 a seed singolo (criterio più facile); seconda lettura informata da P7; C4 non batte il tabulare in CV
RACCOMANDAZIONE:
- Q8 può procedere (dopo approvazione umana) con i 5 candidati, senza cambiare nulla.
- Per «confermato» usare l'AUROC test sulle righe con LVEF noto (clean) e riportare anche tutte le righe; dichiarare che è la seconda lettura del test.
- In Q9 riportare come deboli: «3 derivazioni sufficienti» (piccola perdita reale), «CNN peggiore in alcuni strati» (effetto di composizione), le regole sugli errori.
