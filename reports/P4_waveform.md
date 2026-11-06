# Report P4 · waveform
- Agente: waveform-modeler
- Data: 2026-11-06 14:45
- Stato: OK

## Obiettivo
Addestrare una rete convoluzionale 1D (CNN, rete neurale che scorre lungo il segnale) multi-etichetta sulle forme d'onda ECG a 12 derivazioni, per il composito SHD e per le 11 condizioni, e confrontarla con la baseline tabulare (P1/P2) sulle stesse righe di validation. Promemoria: SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- Handoff P0 (GPU: sì, RTX 5080; N_TRAIN 72475, N_VAL 4626; forme d'onda (N,1,2500,12)), P1 (AUROC val tabulare composito 0.7464 [0.7331-0.7595], hist_gb), P2 (AUROC tabulari per condizione).
- Forme d'onda `EchoNext_{train,val}_waveforms.npy` (memmap, un handle per worker); metadati via `lib.load_data()`. Test mai caricato.
- Ricetta: `legacy_notebooks/04_Modelli_su_forme_d_onda.ipynb`.

## Cosa è stato fatto
- `experiments/P4/train.py` (+ `model_def.py`): ResNet 1D (stem + 4 blocchi residui, pooling medio+massimo, 256 embedding, testa a 12 uscite), loss BCE mascherata (ignora etichette mancanti), AdamW, OneCycle lr 2e-3, batch 128, AMP, seed 42. Input: SOLO forme d'onda (nessuna feature tabulare, `--use-tab` non usato). Checkpoint a ogni epoca in `results/P4/ckpt.pt`, ripresa automatica.
- Smoke test `debug_n=2000`, 2 epoche: loss 0.565 -> 0.345, AUROC composito (prime 2000 righe di val) 0.677 -> 0.705: non casuale.
- Run completo: 12 epoche su tutto il train (72475 ECG), ~5 s/epoca (dati in cache). Si usa l'epoca FINALE (la 12): nessuna scelta dell'epoca sul validation. Per trasparenza l'AUROC di val monitorata per epoca ha picco 0.8307 all'epoca 8 e 0.8232 alla 12 (lieve sovra-adattamento a fine ciclo); il valore riportato è quello conservativo della 12.
- Confronto: modello tabulare hist_gb (7 feature ECG) riaddestrato sul train e valutato sulle stesse righe di val per ciascun target; differenza CNN - tabulare con IC95% bootstrap (300 ripetizioni, sulle stesse righe).
- Stratificazione per fascia di QRS sul composito.

## Risultati
Tutto su validation (1 ECG per paziente). AUROC con IC95% bootstrap.

| Target | n pos | CNN AUROC [IC] | Tabulare AUROC [IC] | Differenza [IC] |
|---|---|---|---|---|
| shd composito | 1990 | 0.823 [0.811-0.834] | 0.746 [0.733-0.759] | +0.077 [0.063, 0.091] |
| lvef_lte_45 | 866 | 0.889 [0.876-0.902] | 0.764 [0.748-0.782] | +0.125 [0.109, 0.141] |
| lvwt_gte_13 | 877 | 0.751 [0.734-0.766] | 0.686 [0.669-0.705] | +0.065 [0.041, 0.084] |
| aortic_stenosis | 252 | 0.831 [0.807-0.851] | 0.844 [0.819-0.868] | -0.012 [-0.039, 0.014] |
| aortic_regurgitation | 62 | 0.768 [0.697-0.818] | 0.632 [0.564-0.694] | +0.136 [0.077, 0.189] |
| mitral_regurgitation | 282 | 0.839 [0.817-0.860] | 0.767 [0.735-0.789] | +0.072 [0.052, 0.103] |
| tricuspid_regurgitation | 305 | 0.843 [0.821-0.865] | 0.753 [0.727-0.779] | +0.090 [0.063, 0.121] |
| pulmonary_regurgitation | 21 | 0.747 [0.602-0.870] | 0.613 [0.500-0.751] | +0.133 [-0.020, 0.271] |
| rv_systolic_dysfunction | 368 | 0.885 [0.868-0.904] | 0.764 [0.737-0.787] | +0.121 [0.098, 0.146] |
| pericardial_effusion | 52 | 0.754 [0.687-0.821] | 0.641 [0.567-0.706] | +0.113 [0.031, 0.203] |
| pasp_gte_45 | 581 | 0.775 [0.755-0.794] | 0.685 [0.663-0.707] | +0.090 [0.064, 0.116] |
| tr_max_gte_32 | 267 | 0.785 [0.756-0.815] | 0.673 [0.642-0.704] | +0.112 [0.073, 0.153] |

Composito per fascia di QRS (val):

| QRS | n | n pos | CNN [IC] | Tabulare [IC] | Differenza |
|---|---|---|---|---|---|
| <100 | 3388 | 1204 | 0.791 [0.774-0.808] | 0.704 [0.689-0.722] | +0.087 |
| 100-119 | 671 | 379 | 0.844 [0.815-0.871] | 0.723 [0.683-0.759] | +0.121 |
| >=120 | 567 | 407 | 0.845 [0.814-0.883] | 0.752 [0.710-0.798] | +0.093 |

GAIN_VS_TABULARE (composito) = +0.077 [0.063, 0.091]. Le forme d'onda portano segnale ben oltre le 7 feature ECG, in tutte le condizioni tranne la stenosi aortica (pari al tabulare), e anche nei QRS larghi. Il CNN è migliore in tutte le fasce di QRS; non emerge una fascia in cui fallisce.

## Decisioni e motivazioni
- Solo forme d'onda (senza feature tabulari) per isolare il contributo del segnale grezzo; il notebook usava anche le feature tabulari, qui scartate per un confronto pulito e per un candidato semplice.
- Epoca finale invece del miglior valore su validation, per non introdurre ottimismo da selezione (costo: ~0.007 di AUROC rispetto al picco).
- Un solo set di iperparametri (nessuna ricerca): ricetta del notebook.
- Nota: il gate P4 prevede `waveform_dominant true` se GAIN >= 0.05 (soddisfatto: 0.077).

## Configurazioni provate
2 configurazioni, entrambe registrate con `log-exp --uses-val` (E0012 smoke test, E0013 run completo). Nessuna scelta di modello o epoca fatta sul validation; il validation è stato usato per valutazione e monitoraggio per epoca. Il modello tabulare di confronto è riaddestrato (hist_gb) come in P1/P2, senza nuove scelte.

## Artefatti
- `experiments/P4/train.py`, `experiments/P4/model_def.py`; log `results/P4_train.log`.
- `results/P4/ckpt.pt` (checkpoint ultima epoca), `results/P4/history.json`, `results/P4/weights.pt`.
- `results/P4/val_per_target.csv`, `results/P4/val_by_qrs.csv`, `results/P4/val_pred_emb.npz` (predizioni 12 target ed embedding 256-dim di validation; solo numeri, non da riportare a livello di paziente).
- Candidato `results/P4/candidates/cnn_multilabel` (target composito; `in_scope` sempre True; carica `weights.pt`, `model_def.py`, `config.json` accanto a sé e legge le forme d'onda con `lib.load_data` + `data.waveforms`). Verificato: riproduce le predizioni salvate.
- Smoke test: `results/P4/debug/`.

## Problemi aperti
- Gli embedding di train non sono stati salvati (solo val); per il clustering facoltativo di P5 servono gli embedding di train, calcolabili con `weights.pt` e `model_def.py`.
- Una sola esecuzione (un seed): nessuna stima della variabilità tra addestramenti.
- Differenza vs tabulare: il tabulare ha solo 7 feature; il confronto dice quanto segnale c'è nelle forme d'onda, non che la CNN sia clinicamente utile. Per pulmonary_regurgitation (21 positivi) e le condizioni rare gli IC sono larghi.
- Il candidato congelabile dovrebbe vincere un test finale su dati nuovi: AUROC di validation, pur senza selezione di epoca, non è una stima indipendente per l'intero processo (smoke test e run sullo stesso validation).
- Il bootstrap è sulle righe (valido: 1 ECG per paziente in val). La scelta del composito come target del candidato è dettata dal playbook.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- CNN_AUROC_VAL_COMPOSITO: 0.823 [0.811-0.834] (AUPRC 0.804; tabulare hist_gb 0.746 [0.733-0.759])
- GAIN_VS_TABULARE: +0.077 [0.063, 0.091] (>= 0.05: waveform_dominant)
- GPU_USATA: sì (RTX 5080)
- TARGET_MIGLIORI: [lvef_lte_45 0.889 [0.876-0.902], rv_systolic_dysfunction 0.885 [0.868-0.904], tricuspid_regurgitation 0.843, mitral_regurgitation 0.839, aortic_stenosis 0.831 (pari al tabulare 0.844)]
- EPOCHE: 12 (usata l'ultima, nessuna selezione; picco monitorato 0.831 all'epoca 8)
- Per QRS il CNN batte il tabulare: <100 0.791, 100-119 0.844, >=120 0.845 (differenze +0.087/+0.121/+0.093)
- CANDIDATO: results/P4/candidates/cnn_multilabel; embedding e predizioni val in results/P4/val_pred_emb.npz
RACCOMANDAZIONE:
- Gate P4: GAIN >= 0.05, impostare waveform_dominant true. P5 può usare gli embedding (rigenerare quelli di train con weights.pt per definire i cluster).
- In P6 segnalare: un solo seed, una sola configurazione, valutazione su validation già usato in P1-P3; ammettere il candidato solo se il confronto nel test finale (P7) regge.
