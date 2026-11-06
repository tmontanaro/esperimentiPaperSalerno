# Report P0 · data-audit
- Agente: data-auditor
- Data: 2026-11-06 14:30
- Stato: OK

## Obiettivo
Verificare che i dati siano utilizzabili prima delle altre fasi: file presenti, split senza pazienti condivisi, schema atteso, valori mancanti, prevalenza delle 12 etichette, forme d'onda coerenti con il CSV, disponibilità di GPU. Nota: l'SHD (cardiopatia strutturale) NON coincide con lo scompenso cardiaco.

## Input letti
- `config.data_dir` = `/tmp/echonext_synth` (cartella indicata dal brief; i nomi suggeriscono dati sintetici, vedi Problemi aperti).
- `lib.load_data()` (train e validation; del test solo conteggi e id aggregati tramite `Data.integrity`).
- Header dei tre file `.npy` (`mmap_mode="r"`, nessun contenuto letto). Dipendenze: nessuna.

## Cosa è stato fatto
- Script `experiments/P0/audit.py` (argomento opzionale `debug_n`). Smoke test con `debug_n=2000`, poi esecuzione completa (che scrive il JSON).
- Controlli: sovrapposizione pazienti, ECG per paziente, colonne attese, frazione di mancanti, prevalenze, quantili di QRS ed età, correlazioni tra le 7 feature (train), shape e N dei `.npy` vs righe CSV, GPU.

## Risultati
**Integrità (conteggi CSV):** train 2127, val 252, test 241 (solo conteggio), più 380 righe con `split = "no_split"`.

| Controllo | Esito |
|---|---|
| Pazienti condivisi train/val, train/test, val/test | 0 / 0 / 0 |
| Max ECG per paziente (train / val / test) | 5 / 1 / 1 |
| Colonne attese (ECG_FEATURES, FLAG_COLS, lvef_value, qrs_duration, patient_key, split) | tutte presenti |
| Forme d'onda train / val / test | shape (N,1,2500,12), N = 2127 / 252 / 241 = righe CSV |
| GPU (`torch.cuda.is_available()`) | True |

**Mancanti (train / val):** le 7 feature sono complete tranne `atrial_rate` e `pr_interval` (8,7% / 8,7%, tipico di ritmi senza onda P misurabile). Le 11 etichette granulari e `lvef_value` mancano nel 18,0% (train) e 21,4% (val) degli ECG, cioè quelli senza eco entro un anno; il composito non ha mancanti. Etichettati: train 1744/2127, val 198/252.

**Prevalenza etichette (positivi sugli etichettati)**

| Etichetta | Train n_pos (prev) | Val n_pos (prev) |
|---|---|---|
| shd_moderate_or_greater_flag | 1374 / 2127 (0,646) | 159 / 252 (0,631) |
| lvef_lte_45_flag | 234 (0,134) | 21 (0,106) |
| lvwt_gte_13_flag | 220 (0,126) | 27 (0,136) |
| aortic_stenosis | 114 (0,065) | 23 (0,116) |
| aortic_regurgitation | 113 (0,065) | 15 (0,076) |
| mitral_regurgitation | 112 (0,064) | 9 (0,045) |
| tricuspid_regurgitation | 124 (0,071) | 11 (0,056) |
| pulmonary_regurgitation | 15 (0,009) | 4 (0,020) |
| rv_systolic_dysfunction | 124 (0,071) | 9 (0,045) |
| pericardial_effusion | 36 (0,021) | 4 (0,020) |
| pasp_gte_45 | 164 (0,094) | 19 (0,096) |
| tr_max_gte_32 | 171 (0,098) | 16 (0,081) |

In validation molte condizioni hanno meno di 20 positivi: gli IC saranno larghi; pulmonary_regurgitation (4 positivi) e pericardial_effusion (4) non sono valutabili in modo affidabile.

**Distribuzioni (train / val):** QRS (ms) mediana 91 / 91; Q1-Q3 78-105 / 77-105; 5°-95° perc. 70-127 / 70-132,9; 99° 145,7 / 144. QRS ≥ 120: 9,6% / 12,7% (pochi pazienti in validation per studi per fascia di QRS: ~32). Età mediana 60 / 60; Q1-Q3 49-72 / 49-70; 5°-95° perc. 34-87 / 27,6-84,5. Maschi 49,8% / 46,4%. Il 5° percentile del QRS coincide col minimo (70 ms): probabile troncamento/valore minimo registrato.

**Correlazioni (train):** `ventricular_rate` e `atrial_rate` r = 0,987 (quasi ridondanti); tutte le altre coppie |r| ≤ 0,065.

## Decisioni e motivazioni
- Stato OK: nessuna sovrapposizione, schema completo, forme d'onda coerenti.
- Le 380 righe `no_split` non sono usate da `lib.load_data` (restano fuori da train/val); non sono state analizzate.
- Per le fasi successive: etichette granulari da trattare come mancanti (non zero) per gli ECG senza eco; considerare la collinearità rate/atrial_rate nei modelli lineari.

## Configurazioni provate
0 configurazioni di modelli; nessun uso del validation per scegliere (solo descrizione). Nessun `log-exp` richiesto.

## Artefatti
- `experiments/P0/audit.py`
- `results/P0/data_profile.json`

## Problemi aperti
- 380 righe con `split = "no_split"` (non previsto dallo schema): non verificata la sovrapposizione di pazienti di queste righe con gli altri split, perché `Data.integrity` considera solo train/val/test. Se contengono pazienti presenti in train/val/test, andrebbero escluse a monte.
- `data_dir` è `/tmp/echonext_synth`: se i dati sono sintetici, le conclusioni cliniche non sono trasferibili; cartella in `/tmp` volatile.
- Prevalenza SHD ~64% molto alta rispetto a una coorte clinica reale.
- torch presente, nessun problema di dipendenze.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- INTEGRITA: OK (sovrapposizioni 0/0/0; ECG per paziente val=1, test=1)
- N_TRAIN: 2127 · N_VAL: 252 (N_TEST solo conteggio: 241)
- PREV_TARGET_TRAIN: 0,646 (1374/2127); validation 0,631 (159/252)
- WAVEFORMS: presenti (shape (N,1,2500,12), N = righe CSV in tutti gli split)
- GPU: sì
- Etichette granulari mancanti: 18,0% train, 21,4% val; atrial_rate/pr_interval mancanti 8,7%
- Positivi in val: lvef_lte_45 21/198; pulmonary_regurgitation 4, pericardial_effusion 4 (non valutabili)
- Avvertenza: 380 righe `no_split` non classificate
RACCOMANDAZIONE:
- Procedere con P1; P4 può addestrare su GPU. Escludere/valutare con cautela condizioni con <10 positivi in validation.
- Verificare l'origine delle righe `no_split` e che i dati non siano solo sintetici.
