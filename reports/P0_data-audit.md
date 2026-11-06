# Report P0 · data-audit
- Agente: data-auditor
- Data: 2026-11-06 12:20
- Stato: PARZIALE

## Obiettivo
Verificare che i dati siano utilizzabili prima di ogni analisi: schema, integrità degli split (nessun paziente condiviso), valori mancanti, prevalenza delle etichette, distribuzioni di QRS ed età, forme d'onda e GPU. Promemoria: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- Nessuna fase precedente (P0 non ha dipendenze).
- Metadati CSV in `config.data_dir` (trovati da `lib.load_data`) e header dei file `EchoNext_{train,val,test}_waveforms.npy`. Del test sono stati usati solo conteggi, id e header dei file.

## Cosa è stato fatto
1. Script `experiments/P0/audit.py` (usa `lib.load_data()`, `Data.integrity`, `lib.show()`).
2. Smoke test con `debug_n=2000`, poi esecuzione completa. Il profilo è scritto solo dal run completo.
3. Forme d'onda: header letto direttamente dal file, confronto N con le righe del CSV, confronto tra dimensione reale e dimensione attesa dal header, tentativo di `mmap_mode="r"`. Il contenuto non è mai stato letto.
4. GPU con `torch.cuda.is_available()`.

## Risultati
**Integrità (conteggi per split dal CSV).**

| Split | Righe ECG | Max ECG per paziente |
|---|---|---|
| train | 72475 | 146 |
| val | 4626 | 1 |
| test | 5442 | 1 |
| no_split | 17457 | non valutato (non usato) |

- Sovrapposizione di pazienti: train–val 0, train–test 0, val–test 0.
- Il train contiene più ECG per paziente (fino a 146): la CV deve essere raggruppata per `patient_key`.
- Tutte le colonne attese sono presenti: ECG_FEATURES, FLAG_COLS, `lvef_value`, `qrs_duration`, `patient_key`, `split`.

**Mancanti (frazione).**

| Colonna | train | val |
|---|---|---|
| atrial_rate | 0.0062 | 0.0067 |
| pr_interval | 0.1051 | 0.1014 |
| lvef_value | 0.0891 | 0.1111 |
| altre 5 feature e 12 flag | 0 | 0 |

- Le 12 etichette `*_flag` non hanno mancanti né in train né in val, quindi non c'è nulla da escludere per l'assenza di eco entro un anno. Non ho verificato come siano codificati gli ECG senza eco (potrebbero essere stati esclusi a monte o messi a 0). Vedi «Problemi aperti».
- Solo `lvef_value` (continuo) ha mancanti: 8.9% in train e 11.1% in val.
- `pr_interval` e `atrial_rate` mancanti vanno gestiti con imputazione o con modelli che li tollerano.

**Prevalenza delle etichette (train n=72475 / val n=4626).**

| Etichetta | train | val | pos. val |
|---|---|---|---|
| shd_moderate_or_greater (target) | 0.5237 | 0.4302 | 1990 |
| lvef_lte_45 | 0.2340 | 0.1872 | 866 |
| lvwt_gte_13 | 0.2438 | 0.1896 | 877 |
| aortic_stenosis | 0.0403 | 0.0545 | 252 |
| aortic_regurgitation | 0.0121 | 0.0134 | 62 |
| mitral_regurgitation | 0.0847 | 0.0610 | 282 |
| tricuspid_regurgitation | 0.1063 | 0.0659 | 305 |
| pulmonary_regurgitation | 0.0083 | 0.0045 | 21 |
| rv_systolic_dysfunction | 0.1324 | 0.0796 | 368 |
| pericardial_effusion | 0.0287 | 0.0112 | 52 |
| pasp_gte_45 | 0.1894 | 0.1256 | 581 |
| tr_max_gte_32 | 0.1034 | 0.0577 | 267 |

- La prevalenza del target è molto diversa tra train (0.524) e val (0.430): le due popolazioni non sono identiche, quindi le metriche di val vanno lette con cautela.
- `pulmonary_regurgitation` ha solo 21 positivi in val, sotto la soglia di 25 del PLAYBOOK: non potrà essere un target «promettente» in val. `aortic_regurgitation` ne ha 62 e `pericardial_effusion` 52.

**QRS ed età (quantili 1/5/25/50/75/95/99%).**
- QRS (ms), train: 64, 70, 80, 90, 102, 142, 168. Val: 62.5, 70, 80, 88, 100, 142, 164.
- Età (anni), train: 23, 30, 52, 63, 73, 85, 90. Val: 22, 30, 52, 64, 75, 88, 90.
- Circa il 5% dei pazienti ha QRS ≥ 142 ms: i gruppi con QRS largo sono piccoli ma non trascurabili.

**Correlazioni tra le 7 feature (Spearman, train).**
- `ventricular_rate` e `atrial_rate`: 0.919, quasi ridondanti.
- `qrs_duration` con `qt_corrected` 0.336, con `pr_interval` 0.296, con `sex` 0.258.
- `pr_interval` con `age` 0.229.
- Le altre coppie hanno |rho| < 0.16.

**Forme d'onda (header vs file).**

| Split | N CSV | N nell'header | Shape | Byte reali / attesi | Righe complete nel file |
|---|---|---|---|---|---|
| train | 72475 | 72475 | (72475,1,2500,12) float64 | 4.39 GB / 17.39 GB | 18299 |
| val | 4626 | 4626 | (4626,1,2500,12) float64 | 0.44 GB / 1.11 GB | 1846 |
| test | 5442 | 5442 | (5442,1,2500,12) float64 | 0.88 GB / 1.31 GB | 3685 |

- I file esistono e l'header è coerente col CSV, ma tutti e tre sono **troncati** (download incompleto). `np.load(mmap_mode="r")` fallisce con «mmap length is greater than file size».
- Righe complete nel file rispetto al CSV: train 25%, val 40%, test 68% (conteggi aggregati, nessun contenuto letto).
- Anche leggendo solo le righe iniziali non si avrebbe un campione rappresentativo, e l'ordine dei file coincide con l'ordine del CSV.

**GPU:** `torch.cuda.is_available()` = True (torch installato).

## Decisioni e motivazioni
- Integrità tabellare: OK. Nessuna sovrapposizione e metadati completi.
- Stato PARZIALE e non FALLITO. Il numero N dell'header coincide con il CSV, quindi non c'è un disallineamento di conteggio. Manca invece il contenuto dei file e solo le forme d'onda sono compromesse. Le fasi tabulari (P1, P2, P3, P5) possono procedere. P4 è bloccata finché i file non vengono riscaricati.
- Alternativa scartata: usare le prime righe complete di ciascun file. Darebbe un sottoinsieme non verificato e non rappresentativo, con il rischio di confondere i risultati.

## Configurazioni provate
0 configurazioni, 0 con uso del validation per scegliere. Nessuna riga registrata con `log-exp`. Sono stati calcolati solo statistiche descrittive su train e validation.

## Artefatti
- `experiments/P0/audit.py`
- `results/P0/data_profile.json` (conteggi, prevalenze, mancanti, quantili, correlazioni, GPU, stato delle forme d'onda)
- Nota in `agentic/DECISIONS.md` (tipo issue) sui file troncati.

## Problemi aperti
- **Forme d'onda troncate (bloccante per P4).** Ri-scaricare `EchoNext_{train,val,test}_waveforms.npy` da PhysioNet (`wget -c` o `-N`), verificare i checksum e rilanciare `python3 experiments/P0/audit.py`: deve dare `WAVEFORMS: presenti`.
- Non ho potuto ispezionare direttamente la cartella dati con la shell (accesso negato dal sandbox). Le dimensioni dei file provengono da `Path.stat()` dentro lo script.
- Le etichette granulari non risultano mancanti. Non ho verificato come siano trattati gli ECG senza eco entro un anno (esclusi a monte o codificati 0): la prevalenza reale potrebbe essere diversa. Da chiarire con la documentazione del dataset.
- Il split `no_split` (17457 righe) non è usato da nessuna fase.
- Il train ha più ECG per paziente (max 146): GroupKFold è obbligatorio, e le righe non sono indipendenti.
- Differenze di prevalenza train/val (es. SHD 0.524 vs 0.430): possibile shift di popolazione.
- `pulmonary_regurgitation` ha 21 positivi in val (< 25): risultati non affidabili.

## Handoff
STATUS: PARZIALE
RISULTATI_CHIAVE:
- INTEGRITA: OK (sovrapposizione pazienti train/val/test = 0/0/0; schema completo)
- N_TRAIN: 72475 (max 146 ECG per paziente, usare GroupKFold)
- N_VAL: 4626 (1 ECG per paziente)
- PREV_TARGET_TRAIN: 0.5237 (val 0.4302, 1990 positivi)
- WAVEFORMS: mancanti (file troncati: righe complete train 18299/72475, val 1846/4626, test 3685/5442)
- GPU: sì
- Mancanti: pr_interval 10.5% train, atrial_rate 0.6%, lvef_value 8.9% train / 11.1% val; flag 0%
- Etichette con pochi positivi in val: pulmonary_regurgitation 21 (< 25)
RACCOMANDAZIONE:
- Procedere con le fasi tabulari (P1, P2, P3, P5): ECG_FEATURES con imputazione per pr_interval, CV raggruppata per patient_key.
- Non lanciare P4 finché i file .npy non vengono riscaricati e P0 rieseguita. Dopo il download la GPU è disponibile per l'addestramento completo.
- Segnalare nei report lo shift di prevalenza train/val.
