# Report Q0 · inventory
- Agente: inventory-auditor
- Data: 2026-11-06 18:00
- Stato: OK

## Obiettivo
Verificare cosa esiste davvero della pipeline 1 e controllare le ipotesi su cui poggia la pipeline 2: ordine delle derivazioni ECG, popolazione «clean» (con LVEF noto), fold per paziente e tempi di calcolo della CNN. Promemoria: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- `python3 tools/state2.py brief` (nessuna dipendenza), `agentic/PLAYBOOK2.md` (Q0 e regole speciali), `agentic/REPORT_TEMPLATE.md`.
- `reports/FINAL_REPORT.md` (sintesi P8), `reports/final_test_table.md`. I numeri della pipeline 1 sono copiati da qui, non ricalcolati.
- Dati: solo pool di sviluppo (train+validation) tramite `pipe2_lib.load_dev`. Il test non è stato caricato.

## Cosa è stato fatto
1. Controllo dell'esistenza degli artefatti della pipeline 1 (sezione Artefatti).
2. `results/Q0/p1_facts.json`: numeri chiave copiati dai report, con la fonte.
3. `experiments/Q0/leads_and_pop.py` (eseguito direttamente sull'intero pool, non in smoke test perché è solo aggregazione): regressione lineare di ogni canale su due altri canali, su 2000 ECG di train scelti a caso (seed 42; 5 milioni di campioni temporali), più conteggi di popolazione clean e fold. Output in `results/Q0/leads_pop.json`.
4. Prova dei tempi: `python3 tools/cnn_cv.py run --out results/Q0/bench --seeds 0 --epochs 2 --time-budget-min 0.01`. Si è fermata dopo il primo job (1/5, codice 3 come previsto); non l'ho rilanciata perché serviva solo la misura.

## Risultati

### Numeri della pipeline 1 (copiati, non ricalcolati)
| Misura | Split | AUROC [IC95%] | Fonte |
|---|---|---|---|
| SHD, CNN | test (n=5442, 2318 pos.) | 0.830 [0.820-0.841] | FINAL_REPORT §2a |
| SHD, CNN, solo righe con LVEF | test (n=4827) | 0.817 [0.807-0.829] | final_test_table |
| SHD, CNN | validation | 0.823 [0.811-0.834] | FINAL_REPORT §3 |
| SHD, tabulare (7 feature) | test | 0.762 [0.750-0.774] | FINAL_REPORT §2a |
| SHD, tabulare | validation (n=4626) | 0.746 [0.733-0.759] | FINAL_REPORT §2a |
| LVEF ≤ 45%, tabulare | test (962 pos.) | 0.793 [0.776-0.809] | FINAL_REPORT §2c |
| LVEF ≤ 45%, testa CNN | validation | 0.889 [0.876-0.902] | FINAL_REPORT §2b |

Prevalenza SHD: train 0.524, validation 0.430, test 0.426. Righe senza LVEF: train 6457 (8.9%), validation 514 (11.1%). Sottogruppi credibili 0, cluster credibili 0; ipotesi QRS inconclusiva. Configurazioni provate: 511 in 22 esperimenti (433 con validation).

### Ordine delle derivazioni (2000 ECG di train)
Regressione con intercetta; canali numerati 0-11.
| Relazione attesa | Coefficienti | R² | Segni attesi |
|---|---|---|---|
| c1 ≈ a·c0 + b·c2 (a,b>0) | 1.024; 0.954 | 0.976 | sì |
| c3 ≈ a·c0 + b·c1 (a,b<0) | −0.574; −0.561 | 0.994 | sì |
| c4 ≈ a·c0 + b·c2 (a>0, b<0) | 0.597; −0.558 | 0.995 | sì |
| c5 ≈ a·c1 + b·c2 (a,b>0) | 0.598; 0.577 | 0.996 | sì |

Tutte e 4 le relazioni hanno R² ≥ 0.95 e segni attesi, con coefficienti vicini a quelli teorici (1, ½). Quindi **LEAD_ORDER_OK = yes** per i canali 0-5 (I, II, III, aVR, aVL, aVF). Le derivazioni toraciche (canali 6-11) NON sono verificabili con questo metodo: la loro identità (V1-V6) resta un'ipotesi non controllata. Il test è condotto su un campione, non ha IC per paziente: è una verifica di struttura del segnale, non una stima di prestazione.

### Popolazione clean e fold (train+validation)
| Quantità | Valore |
|---|---|
| ECG nel pool (N_DEV) | 77101 (train 72475, val 4626) |
| Pazienti | 30844 |
| ECG clean (LVEF noto) | 70130 = 90.96% (train 66018, val 4112) |
| Prevalenza SHD: tutte le righe / clean | 0.518 / 0.570 |

Fold per paziente (ECG / pazienti): 0: 15631/6168; 1: 15883/6260; 2: 15260/6155; 3: 15126/6102; 4: 15201/6159. Bilanciate (differenza massima tra fold circa 2%).
La prevalenza più alta nelle righe clean (0.570 contro 0.518) conferma che le righe senza LVEF sono tutte negative (vedi P6): tenerle gonfia l'AUROC di circa 0.01. Non si può chiarirne il motivo con i dati disponibili.

### Prova dei tempi (GPU, 5 fold del pool intero, 12 derivazioni)
Dispositivo: **cuda (GPU sì)**. Epoca 1: 11 s, epoca 2: 9 s. Job da 2 epoche: 25 s totali, quindi circa 5 s fissi (caricamento, predizione). Si assume SEC_PER_EPOCH ≈ 9.5.
| Stima | Calcolo | Minuti |
|---|---|---|
| 1 job, 12 epoche | 12×9.5+5 s | ≈ 2.0 |
| Q1: 5 fold × 3 seed = 15 job | 15×2.0 | ≈ 30 |
| Q2: 7 set di derivazioni × 5 fold = 35 job | 35×2.0 (limite superiore: con meno derivazioni è più veloce) | ≤ 70 |
| Q6: final-fit 3 seed sul pool intero (circa 1.25× dati) | 3×2.5 | ≈ 8 |

Sono stime da un solo job, senza IC: la variabilità del tempo non è misurata. Q1 resta ampiamente sotto le 3 ore, quindi l'esplorazione limitata prevista in Q1 (3 varianti con seed 0) è consentita.

## Decisioni e motivazioni
- Ordine delle derivazioni verificato (`yes`): nei report successivi si possono citare i nomi delle derivazioni 0-5; per 6-11 no, finché non verificato.
- Popolazione principale = clean, con *all* come sensibilità (regola 4 del PLAYBOOK2).
- Benchmark fermato dopo il primo job, come previsto; non rilanciato per risparmiare tempo.
- Alternativa scartata: ricalcolare i numeri della pipeline 1. Il PLAYBOOK chiede di copiarli.

## Configurazioni provate
0 nuove configurazioni. Nessuna scelta è stata fatta con il validation; il benchmark misura solo i tempi, quindi non ho usato `log-exp`. Totale ereditato dal brief: 511 (22 esperimenti).

## Artefatti
- `results/Q0/p1_facts.json`, `results/Q0/leads_pop.json`, `results/Q0/bench/` (progress e job di prova), `experiments/Q0/leads_and_pop.py`.
- Esistono: `results/P4/weights.pt` e `results/P4/candidates/cnn_multilabel` (pesi CNN), `results/final/test_results.json`, `agentic/FROZEN.json`, `results/registry.csv` (22 righe di dati + intestazione), `results/P1/candidates/global_best`, `results/P3/candidates/lvef_lte45_global`.

## Problemi aperti
- Derivazioni toraciche (canali 6-11) non verificabili.
- Il numero di derivazioni verificate dipende da un campione di 2000 ECG; l'R² di c1 (0.976) è il più basso, ma sopra soglia.
- Il significato delle righe senza LVEF resta ignoto.
- Tempi stimati da un solo job; Q2 è un limite superiore.
- Non ho riletto `results/registry.csv` riga per riga: ho solo contato le righe (22 esperimenti, coerente col brief). Ho preso 511 e 433 dal brief e da P8.
- `reports/P0_data-audit.md` e i report P1-P7 singoli non sono stati letti: mi sono basato sulla sintesi P8.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- LEAD_ORDER_OK: yes (canali 0-5; toraciche non verificabili)
- R2_LEAD_RELATIONS: c1 0.976, c3 0.994, c4 0.995, c5 0.996 (segni attesi)
- N_DEV: 77101 ECG, 30844 pazienti; CLEAN_FRACTION: 0.910 (70130 ECG)
- SEC_PER_EPOCH: 9.5 (GPU: sì)
- EST_MIN_Q1: 30; EST_MIN_Q2: 70 (limite superiore); EST_MIN_Q6: 8
- P1_ARTEFATTI_MANCANTI: [] (pesi CNN, test_results.json, candidati P1/P3/P4, registry presenti)
- Riferimenti P1: SHD CNN test 0.830 [0.820-0.841], tabulare 0.762 [0.750-0.774]
RACCOMANDAZIONE:
- Q1 può partire: circa 30 min, quindi l'esplorazione di 3 varianti è ammessa. Usare la popolazione clean e fold per paziente.
- Non nominare le derivazioni toraciche nei report.
