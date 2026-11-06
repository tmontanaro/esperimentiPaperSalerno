# Report P0 · data-audit
- Agente: data-auditor
- Data: 2026-11-06 14:30
- Stato: OK

## Obiettivo
Verificare che i dati EchoNext siano utilizzabili: split senza pazienti condivisi, colonne attese, mancanti, forme d'onda coerenti col CSV, profilo delle 12 etichette. Serve a tutte le fasi successive. Promemoria: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- Config `data_dir` = /home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1 (CSV di metadati e file `EchoNext_{train,val,test}_waveforms.npy`).
- Nessun handoff (P0 non ha dipendenze). Del test sono stati usati solo conteggi, id aggregati e l'header dei file .npy.

## Cosa è stato fatto
- Script `experiments/P0/audit.py` (usa `lib.load_data()`, `Data.integrity`, `lib.show()`).
- Smoke test con `debug_n=2000` (output in `results/P0/debug_profile.json`), poi run completo (`results/P0/data_profile.json`).
- Controlli: overlap pazienti, ECG per paziente, colonne attese, mancanti, prevalenze, quantili di QRS ed età, correlazioni, header .npy via `mmap_mode="r"`, GPU.

## Risultati
Conteggi per split (ECG): train 72475, val 4626, test 5442, no_split 17457 (scartati dalla libreria). Pazienti: train 26218, val 4626.
Sovrapposizione pazienti: train-val 0, train-test 0, val-test 0. ECG per paziente: massimo 1 in val e test; in train massimo 146 (12023 pazienti su 26218 hanno più di un ECG; mediana 1, 90° percentile 5, 99° percentile 23). Quindi la CV in train va SEMPRE raggruppata per `patient_key`.
Colonne attese: tutte presenti (ECG_FEATURES, FLAG_COLS, `lvef_value`, `qrs_duration`, `patient_key`, `split`).

Mancanti (frazione, train / val): `pr_interval` 0.105 / 0.101; `atrial_rate` 0.0062 / 0.0067; `lvef_value` 0.089 / 0.111; tutte le altre feature e le 12 etichette 0.000. Le etichette granulari non risultano mancanti in nessuna riga di train/val (nessuna riga con etichetta NaN): gli ECG senza eco entro un anno non sono presenti in questi split (o sono tra i `no_split`, non analizzati). Il mancante di `lvef_value` (solo valore continuo, 8.9% / 11.1%) va gestito in P3.

Prevalenza etichette (train n=72475 / val n=4626):

| Etichetta | Prev. train | n pos train | Prev. val | n pos val |
|---|---|---|---|---|
| shd_moderate_or_greater (target) | 0.524 | 37958 | 0.430 | 1990 |
| lvef_lte_45 | 0.234 | 16962 | 0.187 | 866 |
| lvwt_gte_13 | 0.244 | 17667 | 0.190 | 877 |
| aortic_stenosis | 0.040 | 2919 | 0.055 | 252 |
| aortic_regurgitation | 0.012 | 878 | 0.013 | 62 |
| mitral_regurgitation | 0.085 | 6137 | 0.061 | 282 |
| tricuspid_regurgitation | 0.106 | 7707 | 0.066 | 305 |
| pulmonary_regurgitation | 0.008 | 603 | 0.0045 | 21 |
| rv_systolic_dysfunction | 0.132 | 9597 | 0.080 | 368 |
| pericardial_effusion | 0.029 | 2079 | 0.011 | 52 |
| pasp_gte_45 | 0.189 | 13727 | 0.126 | 581 |
| tr_max_gte_32 | 0.103 | 7492 | 0.058 | 267 |

Le prevalenze in validation sono sistematicamente più basse che in train (train conta più ECG per paziente, quindi pazienti malati pesano di più): va ricordato nel confronto train/val. Etichette rare in validation (pulmonary_regurgitation 21 positivi, pericardial_effusion 52, aortic_regurgitation 62) avranno IC molto larghi.

Quantili (1%,5%,25%,50%,75%,95%,99%): QRS train 64, 70, 80, 90, 102, 142, 168 ms; QRS val 62.5, 70, 80, 88, 100, 142, 164. Età train 23, 30, 52, 63, 73, 85, 90; età val 22, 30, 52, 64, 75, 88, 90 (età troncata a 90). Frazione di sesso maschile: vedi `data_profile.json`.

Correlazioni tra feature (train): massime ventricular_rate-atrial_rate 0.49, qrs-qt_corrected 0.40, pr-qrs 0.28, pr-age 0.23, sex-qrs 0.21; le altre |r| < 0.30. Nessuna collinearità problematica.

Forme d'onda (solo header): train (72475,1,2500,12), val (4626,1,2500,12), test (5442,1,2500,12); N coincide col CSV in tutti e tre gli split.
GPU: `torch.cuda.is_available()` = True (NVIDIA GeForce RTX 5080, torch 2.15.0+cu132).

## Decisioni e motivazioni
Esito OK: nessuna sovrapposizione, metadati completi, forme d'onda coerenti. Le righe `no_split` non sono usate. Nessuna feature scartata in questa fase.

## Configurazioni provate
0 configurazioni di modelli; 0 hanno usato il validation per scegliere (audit descrittivo).

## Artefatti
- /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/experiments/P0/audit.py
- /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P0/data_profile.json
- /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P0/debug_profile.json

## Problemi aperti
- Etichette granulari mai NaN in train/val: non è possibile quantificare gli ECG "senza eco entro un anno" da questi split; probabilmente filtrati a monte.
- Molti ECG per paziente in train (max 146): rischio di sovrastimare la precisione senza GroupKFold; e pazienti molto rappresentati dominano.
- Differenza di prevalenza train vs val (target 0.524 vs 0.430) = possibile shift di distribuzione.
- `pr_interval` mancante nel ~10%: imputare dentro la pipeline (in CV).
- Le 17457 righe `no_split` non sono state valutate.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- INTEGRITA: OK (overlap pazienti train/val/test = 0; max 1 ECG per paziente in val e test, max 146 in train)
- N_TRAIN: 72475 ECG (26218 pazienti); N_VAL: 4626 ECG (4626 pazienti)
- PREV_TARGET_TRAIN: 0.524 (val 0.430)
- WAVEFORMS: presenti (shape (N,1,2500,12), N coincide col CSV in train/val/test)
- GPU: sì (RTX 5080)
- Mancanti: pr_interval 10.5%, atrial_rate 0.6%, lvef_value 8.9% (train)
RACCOMANDAZIONE:
- Usare GroupKFold su `patient_key` in train; imputare pr_interval/atrial_rate nella pipeline.
- P3: gestire i NaN di lvef_value (escluderli, dichiarando n); P4: GPU disponibile, addestramento completo possibile.
