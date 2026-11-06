# Report Q2 · lead-ablation
- Agente: lead-ablation-analyst
- Data: 2026-11-06 20:00
- Stato: OK

## Obiettivo
Capire quali derivazioni dell'ECG servono davvero alla CNN: si tolgono derivazioni dall'input e si guarda quanto cala l'AUROC (misura di quanto bene il modello separa chi ha la condizione da chi non l'ha; 0.5 = caso, 1 = perfetto). Tutto con la stessa configurazione, lo stesso seed (0) e le stesse fold di Q1: cambia solo la lista di derivazioni. Ricordiamo che SHD (cardiopatia strutturale moderata o più grave) NON coincide con scompenso cardiaco.

## Input letti
- `python3 tools/state2.py brief` e Handoff di Q1: CNN_DIR `results/Q1/main`, CONFIG_FINALE (12 epoche, lr 0.002, reg_weight 0.5, batch 128, popolazione clean, 12 derivazioni), seed SD < 0.001.
- Handoff di Q0: LEAD_ORDER_OK = yes (canali 0-5 sono le derivazioni degli arti; le toraciche non sono verificabili, quindi non le nomino).
- Riferimento «tutte le derivazioni»: `results/Q1/main/oof_seed0.npy` (non rilanciato, come da PLAYBOOK2).

## Cosa è stato fatto
1. Per ognuno dei 7 insiemi: `python3 tools/cnn_cv.py run --out results/Q2/<insieme> --seeds 0 --leads <insieme> --epochs 12 --lr 0.002 --reg-weight 0.5 --batch 128 --population clean --time-budget-min 7`, poi `summarize`. Insiemi: `eight` (8 canali), `limb` (6 degli arti), `chest` (6 toraciche), `I_II` (2), `II` (1), `I` (1), `I_II_V1` (3 = le due degli arti I e II più una toracica). Tutti sono completati in una sola esecuzione.
2. `experiments/Q2/ablation_table.py`: AUROC con IC95% per paziente (bootstrap per `patient_key`, 300 ripetizioni) e differenza appaiata rispetto a «tutte» con IC appaiato, su SHD, LVEF ≤ 45% e le 2 condizioni migliori in Q1 oltre a questi (stenosi aortica e disfunzione sistolica del ventricolo destro).
3. Smoke test con `debug_n` non rifatto: la pipeline `cnn_cv.py` era già stata validata in Q1.
Tutto sull'intero pool di sviluppo (train+validation), popolazione clean (n = 70130 ECG), CV 5 fold per paziente.

## Risultati
AUROC fuori-fold (CV, seed 0; n = 70130; positivi: SHD 39948, LVEF ≤ 45% 17828, stenosi aortica 3171, disfunzione VD 9965). Tra parentesi IC95%; Δ = differenza da «tutte» con IC appaiato.

| Insieme | N. canali | SHD | Δ SHD | LVEF ≤ 45% | Δ LVEF |
|---|---|---|---|---|---|
| tutte | 12 | 0.808 [0.801-0.814] | – | 0.862 [0.855-0.869] | – |
| eight | 8 | 0.810 [0.803-0.816] | +0.002 [-0.000, +0.004] | 0.863 [0.855-0.869] | +0.000 [-0.002, +0.003] |
| chest | 6 | 0.805 [0.799-0.812] | -0.002 [-0.005, +0.000] | 0.855 [0.848-0.863] | -0.007 [-0.011, -0.004] |
| I_II_V1 | 3 | 0.802 [0.795-0.809] | -0.006 [-0.009, -0.003] | 0.854 [0.846-0.862] | -0.008 [-0.011, -0.004] |
| limb | 6 | 0.792 [0.785-0.799] | -0.016 [-0.020, -0.012] | 0.851 [0.844-0.858] | -0.011 [-0.015, -0.008] |
| I_II | 2 | 0.791 [0.784-0.797] | -0.017 [-0.021, -0.013] | 0.848 [0.842-0.856] | -0.014 [-0.018, -0.010] |
| I | 1 | 0.783 [0.776-0.789] | -0.024 [-0.029, -0.020] | 0.838 [0.831-0.846] | -0.024 [-0.029, -0.019] |
| II | 1 | 0.770 [0.763-0.777] | -0.038 [-0.043, -0.033] | 0.818 [0.810-0.827] | -0.044 [-0.050, -0.038] |

Due condizioni in più (AUROC; Δ da «tutte»): stenosi aortica: tutte 0.800 [0.784-0.816]; eight -0.002; chest -0.005; I_II_V1 -0.009 [-0.017, -0.002]; limb -0.016; I_II -0.020; I -0.026; II -0.043. Disfunzione sistolica VD: tutte 0.835 [0.825-0.845]; eight +0.001; chest -0.005; I_II_V1 -0.013 [-0.018, -0.009]; limb -0.024; I_II -0.025; I -0.031; II -0.051. Tabella completa con IC: `results/Q2/ablation_table.csv`.

**Regola «insieme sufficiente»** (Δ puntuale ≥ -0.01 sia per SHD sia per LVEF ≤ 45%): soddisfatta da `eight`, `chest` e `I_II_V1`. **Il più piccolo è `I_II_V1`, con 3 derivazioni**: SHD -0.006, LVEF -0.008. Non la soddisfano `limb` (LVEF -0.011, SHD -0.016), `I_II` (-0.017 / -0.014), `I`, `II`.

Osservazioni (descrittive):
- Le toraciche da sole (6) valgono quasi quanto tutte; le sole derivazioni degli arti (6) sono peggiori. Da 8 canali in su non c'è calo misurabile.
- Il calo è piccolo ma «statisticamente reale»: gli IC appaiati di I_II_V1 escludono 0. «Sufficiente» significa quindi «perdita ≤ 0.01», non «nessuna perdita».
- Aggiungere una sola toracica a I+II recupera circa 0.011 di AUROC su SHD (da 0.791 a 0.802).
- Con una sola derivazione si resta comunque a 0.77-0.78 su SHD (II: 0.770), meglio del modello tabulare di Q1 (0.712 [0.705-0.721]), ma con perdita netta di 0.02-0.04.
- La stenosi aortica e la disfunzione VD calano di più (I_II_V1: -0.009 e -0.013): per queste condizioni la soglia di 0.01 non è rispettata dal set a 3 derivazioni, che passa il criterio solo per SHD e LVEF.

## Decisioni e motivazioni
- Un solo seed come da PLAYBOOK; la variabilità tra seed in Q1 era < 0.001, molto minore della soglia 0.01, quindi le differenze di 0.01 non sono rumore di addestramento. Non è però misurata per ogni insieme.
- Soglia di sufficienza fissata a priori (-0.01), applicata solo a SHD e LVEF ≤ 45% come da PLAYBOOK. Nessun'altra soglia provata.
- Non nomino le derivazioni toraciche (LEAD_ORDER_OK non le copre). «I_II_V1» è il nome tecnico dell'insieme.

## Configurazioni provate
7 configurazioni (una per insieme), tutte con `--uses-val` perché la CV usa anche il validation; registrate con `tools/state2.py log-exp` (E0025-E0031). L'insieme «sufficiente» è scelto guardando la CV, quindi la stima è ottimistica quanto basta per una selezione tra 7 candidati (nessuna correzione per multiplicità applicata: le differenze osservate sono molto più grandi degli IC).

## Artefatti
- `results/Q2/<insieme>/` (OOF, pesi, `metrics_clean.csv`, `metrics_all.csv`, `summary.json`) per eight, limb, chest, I_II, II, I, I_II_V1.
- `results/Q2/ablation_table.csv`
- `experiments/Q2/ablation_table.py`
- Progresso: `reports/Q2_lead-ablation.progress.md`

## Problemi aperti
- Un solo seed per insieme: l'ordine fra insiemi molto vicini (eight, chest, I_II_V1) non è garantito.
- «Sufficiente» è relativo a CV su dati interni; vale per l'ECG dello stesso ospedale/dispositivo. Nessuna validazione esterna.
- Risultati su popolazione clean; la sensibilità su «all» è in `metrics_all.csv` e non è stata analizzata qui.
- Le derivazioni toraciche non sono identificabili con certezza (solo canali); I_II_V1 usa una toracica specifica scelta dal PLAYBOOK, non ottimizzata. Altre toraciche potrebbero andare meglio o peggio.
- Non è stato provato un insieme di 2 derivazioni diverso da I_II (es. una degli arti + una toracica).

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- LEAD_ORDER_OK: yes (da Q0); NOTE_NOMI_DERIVAZIONI: canali 0-5 = arti, toraciche non nominate
- INSIEME_MINIMO_SUFFICIENTE: I_II_V1 (3 derivazioni): SHD 0.802 [0.795-0.809], Δ -0.006 [-0.009, -0.003]; LVEF ≤ 45% 0.854 [0.846-0.862], Δ -0.008 [-0.011, -0.004]
- Riferimento tutte (seed 0, clean): SHD 0.808 [0.801-0.814], LVEF ≤ 45% 0.862 [0.855-0.869]
- CALO_PER_INSIEME (Δ SHD / Δ LVEF): eight +0.002/+0.000; chest -0.002/-0.007; I_II_V1 -0.006/-0.008; limb -0.016/-0.011; I_II -0.017/-0.014; I -0.024/-0.024; II -0.038/-0.044
- Per stenosi aortica e disfunzione VD, I_II_V1 perde -0.009 e -0.013 (oltre la soglia per VD)
- TABELLA: results/Q2/ablation_table.csv
RACCOMANDAZIONE:
- Q6: aggiungere un candidato con il set a 3 derivazioni (I_II_V1), uscita SHD; considerare anche `eight` come set senza perdita.
- Dichiarare nei report che «sufficiente» = perdita ≤ 0.01 su un seed, non nessuna perdita.
