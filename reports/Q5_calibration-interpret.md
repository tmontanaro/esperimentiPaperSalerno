# Report Q5 · calibration-interpret
- Agente: calibration-analyst
- Data: 2026-11-06 21:00
- Stato: OK

## Obiettivo
Verificare se le probabilità della CNN sono affidabili (calibrazione: «se la rete dice 70%, nel 70% dei casi la condizione è presente?»), proporre soglie operative con i loro limiti e mostrare quali derivazioni e quali istanti dell'ECG pesano di più sulla predizione. Ricordiamo che SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco. Tutti i numeri sono aggregati.

## Input letti
- `python3 tools/state2.py brief` e Handoff di Q1: CNN_DIR = `results/Q1/main`, AUROC SHD clean 0.815 [0.809-0.822]; TARGETS_GOOD = shd, lvef_lte_45, rv_systolic_dysfunction, aortic_stenosis; config: 12 epoche, lr 0.002, reg_weight 0.5, 12 derivazioni, popolazione clean.
- `results/Q1/main/oof_ensemble.npy` (probabilità fuori-fold, media di 3 seed), pesi `weights_seed0_fold{0,1}.pt`, `oof_seed0_fold*.npz`.
- Q0: LEAD_ORDER_OK = yes solo per i canali 0-5 (arti). Le derivazioni toraciche non sono verificabili e qui sono indicate solo come «canale N».
- `agentic/PLAYBOOK2.md`, sezione Q5.

## Cosa è stato fatto
Tutti gli script sono in `experiments/Q5/`, sull'intero pool di sviluppo (train+validation, popolazione clean, 70 130 ECG). Non sono stati fatti smoke test con `debug_n` separati: gli script sono rapidi e lavorano su file già prodotti da Q1. Il test set non è stato toccato.
1. `calib_ops.py` (lanciato tramite `run_calib.py`, che ha corretto i nomi di due etichette: la prima esecuzione le aveva saltate, poi è stata rifatta per intero): tabelle di affidabilità a 10 bin, ECE (errore medio tra probabilità prevista e frequenza osservata) e Brier (errore quadratico medio delle probabilità) con IC95% bootstrap per paziente (200 ripetizioni), per SHD, LVEF ≤ 45%, disfunzione sistolica del ventricolo destro e stenosi aortica. Ricalibrazione isotonica *fuori dal fold*: per ogni fold, la mappa è appresa sulle predizioni OOF degli altri 4 fold e applicata al fold lasciato fuori.
2. Punti operativi per SHD e LVEF ≤ 45%: soglia per sensibilità 90%, per specificità 90% e per massimo Youden (sensibilità + specificità − 1), con sensibilità, specificità, PPV, NPV e IC per paziente. Le soglie sono scelte sugli stessi dati OOF su cui sono valutate: i valori sono ottimistici per costruzione, ma i punti sono pochi (3 per etichetta) e i campioni grandi.
3. `occlusion.py <fold>`: modello (seed 0, fold 0 e poi fold 1) caricato con `cnn_cv.load_fold_model`; 5000 ECG clean del fold di verifica scelti a caso (seed 42; più ECG possono venire dallo stesso paziente, quindi IC per paziente). Si azzera una derivazione alla volta (12 prove) e poi una finestra da 0.5 s alla volta su tutte le derivazioni (10 prove, ECG da 5 s) e si misura la variazione di AUROC rispetto allo stesso modello senza occlusione (differenza appaiata, IC per paziente). `summarize.py` raccoglie i numeri.

## Risultati
Tutto è calcolato in cross-validation per paziente (OOF) sul pool train+validation; non è il test.

**a) Calibrazione (probabilità non ricalibrate = «grezze»)**

| Etichetta | Prevalenza | ECE grezza | ECE isotonica | Brier grezza | Brier isotonica | Pendenza di calibrazione |
|---|---|---|---|---|---|---|
| SHD | 0.570 (39 948 pos.) | 0.031 [0.026-0.036] | 0.003 [0.003-0.009] | 0.175 [0.171-0.177] | 0.174 [0.170-0.176] | 0.81 |
| LVEF ≤ 45% | 0.254 (17 828) | 0.034 [0.029-0.038] | 0.002 [0.003-0.009] | 0.115 [0.111-0.118] | 0.113 [0.109-0.116] | 0.76 |
| Disfunzione VD | 0.142 (9 965) | 0.016 [0.012-0.021] | 0.002 [0.002-0.008] | 0.090 [0.086-0.095] | 0.090 [0.086-0.095] | 0.83 |
| Stenosi aortica | 0.045 (3 171) | 0.004 [0.002-0.007] | 0.001 [0.002-0.005] | 0.039 [0.037-0.042] | 0.040 [0.037-0.042] | 0.90 |

- Lettura: la rete è abbastanza calibrata (errore medio 0.2-3.4 punti percentuali) ma un po' «troppo sicura» (pendenza < 1: le probabilità alte sono leggermente troppo alte, quelle basse troppo basse). L'AUROC non cambia con la ricalibrazione (SHD 0.815 → 0.814; LVEF ≤ 45% 0.869 → 0.868), come atteso.
- La ricalibrazione isotonica riduce l'ECE di circa 10 volte, ma il guadagno nel Brier è piccolo (SHD −0.0009, LVEF −0.0016: gli IC si sovrappongono quasi del tutto). Per stenosi aortica il Brier non migliora. Nota tecnica: l'ECE bootstrap ha un piccolo bias verso l'alto, per questo i valori puntuali isotonici (0.001-0.003) possono cadere sotto il proprio IC.
- **Shift di prevalenza**: nei dati di addestramento clean la prevalenza di SHD è 0.575, nel validation 0.484. Sul sottoinsieme validation (4 112 ECG) la rete sovrastima: probabilità media 0.518 contro frequenza osservata 0.484 (ECE 0.040 [0.032-0.055] contro 0.031 in train). Per LVEF ≤ 45% 0.224 contro 0.211 (ECE 0.020); per stenosi aortica la prevalenza va in senso opposto (validation 0.061 osservata contro 0.045 prevista, ECE 0.017). La calibrazione assoluta dipende quindi dalla popolazione: in un ospedale con altra prevalenza va ricalibrata. Il confronto train/validation non è un test di significatività.

**b) Punti operativi (OOF, tutti i dati clean; IC95% per paziente)**

| Etichetta | Criterio | Soglia | Sensibilità | Specificità | PPV | NPV | % ECG segnalati |
|---|---|---|---|---|---|---|---|
| SHD | sens. 90% | 0.296 | 0.900 [0.896-0.906] | 0.446 [0.437-0.455] | 0.683 [0.673-0.692] | 0.771 [0.763-0.782] | 75.1% |
| SHD | spec. 90% | 0.739 | 0.542 [0.532-0.554] | 0.900 [0.894-0.907] | 0.878 [0.869-0.886] | 0.598 [0.589-0.609] | 35.2% |
| SHD | Youden | 0.586 | 0.682 [0.673-0.693] | 0.799 [0.790-0.806] | 0.818 [0.809-0.827] | 0.655 [0.646-0.665] | 47.5% |
| LVEF ≤ 45% | sens. 90% | 0.073 | 0.900 [0.891-0.909] | 0.573 [0.563-0.580] | 0.418 [0.406-0.432] | 0.944 [0.938-0.949] | 54.7% |
| LVEF ≤ 45% | spec. 90% | 0.372 | 0.669 [0.655-0.684] | 0.900 [0.895-0.905] | 0.695 [0.681-0.711] | 0.888 [0.883-0.894] | 24.5% |
| LVEF ≤ 45% | Youden | 0.228 | 0.755 [0.742-0.768] | 0.835 [0.829-0.842] | 0.609 [0.594-0.626] | 0.909 [0.904-0.914] | 31.5% |

- Limiti: PPV e NPV valgono solo con la prevalenza di questa popolazione (SHD 0.57, LVEF ≤ 45% 0.25). Sono pazienti che hanno fatto un'ecocardiografia entro un anno dall'ECG: una popolazione già selezionata (bias di selezione), non lo screening della popolazione generale. In una popolazione con prevalenza minore il PPV scenderebbe molto con la stessa sensibilità/specificità. Le soglie sono sulla scala delle probabilità grezze (media di 3 seed): se si ricalibra, vanno ricalcolate. Per SHD, a sensibilità 90% la specificità è solo 0.446: la rete non permette di escludere SHD con confidenza senza segnalare quasi tre ECG su quattro. Per LVEF ≤ 45% invece l'NPV a sensibilità 90% è 0.944, utile per «escludere» in questa popolazione.

**c) Cosa guarda la rete (occlusione; variazione di AUROC, IC95% per paziente)**
Base fold 0: SHD 0.813 [0.795-0.830], LVEF ≤ 45% 0.854 [0.833-0.874] (5000 ECG per fold). Base fold 1: SHD 0.801 [0.783-0.818], LVEF ≤ 45% 0.856 [0.837-0.876].

| Occlusione | SHD fold 0 | SHD fold 1 | LVEF≤45 fold 0 | LVEF≤45 fold 1 |
|---|---|---|---|---|
| Canale 6 (toracico) | −0.026 [−0.035, −0.018] | −0.013 [−0.022, −0.003] | −0.023 [−0.030, −0.015] | −0.011 [−0.020, −0.003] |
| Canale 0 (I) | −0.006 [−0.011, −0.001] | −0.008 [−0.013, −0.003] | −0.006 [−0.011, −0.000] | −0.007 [−0.012, −0.000] |
| Canale 1 (II) | −0.008 [−0.012, −0.004] | −0.002 [−0.006, 0.002] | +0.001 [−0.004, 0.005] | −0.007 [−0.011, −0.003] |
| Canale 9 (toracico) | −0.008 [−0.012, −0.004] | −0.005 [−0.009, −0.001] | −0.004 [−0.008, 0.001] | −0.005 [−0.009, −0.001] |
| Canale 3 (aVR) | −0.006 [−0.009, −0.002] | −0.004 [−0.008, 0.000] | −0.004 [−0.009, −0.001] | −0.007 [−0.010, −0.003] |
| Tutte le altre derivazioni | da −0.005 a +0.003 | da −0.007 a −0.001 | da −0.008 a +0.003 | da −0.007 a −0.002 |

- L'unica derivazione che si distingue è il canale 6 (toracico): calo di 0.013-0.026 per SHD e 0.011-0.023 per LVEF ≤ 45%, con segno e primo posto uguali nei due fold (la grandezza è diversa tra fold). Tutte le altre derivazioni, tolte da sole, fanno calare l'AUROC di meno di 0.01: l'informazione è molto ridondante tra derivazioni (coerente con Q2, dove 3 derivazioni bastano con calo ≈ 0.006-0.008).
- Finestre temporali da 0.5 s: ogni finestra fa calare l'AUROC al massimo di 0.004 per SHD (fold 0: massimo 0.0025 a 3.0-3.5 s; fold 1: massimo 0.0039 a 1.5-2.0 s) e al massimo di 0.006 per LVEF ≤ 45% (IC più stretti tra −0.009 e 0.000). Nessuna finestra si distingue: la rete usa l'ECG in modo diffuso su tutti i 5 secondi, non un battito particolare. Sono variazioni minime e vicine al rumore dell'IC.
- Coerenza fra fold: la correlazione di rango (Spearman) tra i calo dei due fold è 0.20 (derivazioni, SHD), 0.26 (derivazioni, LVEF), −0.02 e 0.36 (finestre): bassa e non significativa (p da 0.31 a 0.96). La coerenza vale quindi **solo per il messaggio principale** (canale 6 più importante, effetti per tutte le altre piccoli e distribuiti), non per l'ordine fine delle derivazioni o delle finestre, che è rumore (differenze ≤ 0.005 contro IC ± 0.004).

## Decisioni e motivazioni
- Ricalibrazione utile? **Sì, ma in modo marginale**: abbatte l'ECE (0.031 → 0.003 per SHD) senza cambiare la discriminazione e con un Brier quasi invariato; è indispensabile soprattutto se si usa la probabilità come numero assoluto o in un'altra popolazione. Per la sola classificazione con soglia non serve.
- Soglie: le tre soglie sono sulla probabilità grezza e derivano dall'OOF del pool di sviluppo; non sono raccomandazioni cliniche, solo punti operativi con i loro costi. Alternative scartate: soglie scelte sul validation (troppo piccolo, 4 112 ECG, e già usato).
- Occlusione per azzeramento: scelta per semplicità e per rispettare il Playbook; scartati metodi a gradiente (non richiesti e più difficili da interpretare).
- Nessuna derivazione toracica è nominata (ordine non verificato da Q0).

## Configurazioni provate
Registrate con `tools/state2.py log-exp` (tutte `--uses-val`: la CV usa anche il validation): E0036 calibrazione (8: 4 etichette × grezza/isotonica), E0037 punti operativi (6), E0038 occlusione (44: 22 occlusioni × 2 fold). Totale 58. Nessuna è stata usata per scegliere un modello: sono analisi descrittive.

## Artefatti
- `results/Q5/calib_ops.json` (calibrazione e punti operativi con IC), `results/Q5/reliability.csv` (tabelle di affidabilità a 10 bin, grezze e isotoniche), `results/Q5/occlusion_fold0.json`, `results/Q5/occlusion_fold1.json`.
- Script: `experiments/Q5/calib_ops.py`, `run_calib.py`, `occlusion.py`, `summarize.py`.

## Problemi aperti
- Le soglie e i PPV/NPV sono valutati sugli stessi dati OOF in cui sono scelti, e valgono per la prevalenza di questo campione (pazienti con eco, un solo centro, nessuna validazione esterna). Non c'è un confronto di calibrazione sul test (chiuso; lo vedrà Q8 per i candidati).
- L'occlusione azzera un segnale che la rete non ha mai visto azzerato (fuori distribuzione) e si fa su un solo modello (seed 0) in 2 fold su 5, con 5000 ECG ciascuno. Un calo piccolo non prova che una derivazione sia inutile (ridondanza); un calo grande non è causale. L'ordine delle derivazioni toraciche non è verificato, quindi il «canale 6» non è collegato a un nome.
- Le confronti di occlusione (44) non sono corretti con Holm: sono descrittivi e si usa il criterio IC che esclude 0 come indicazione, non come prova.
- La ricalibrazione isotonica usa predizioni OOF generate da modelli diversi per fold (e media di 3 seed); in pratica è una stima ragionevole, non esatta, per un modello finale addestrato su tutto il pool.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- ECE_SHD: 0.031 [0.026-0.036] grezza, 0.003 [0.003-0.009] isotonica (Brier 0.175 → 0.174); ECE_LVEF: 0.034 [0.029-0.038] → 0.002 [0.003-0.009] (Brier 0.115 → 0.113); VD 0.016, stenosi aortica 0.004; pendenza 0.76-0.90 (leggermente troppo sicura)
- RICALIBRAZIONE_UTILE: si (solo per il valore assoluto delle probabilità: ECE ÷10, Brier quasi invariato, AUROC invariata); shift di prevalenza: SHD train 0.575 contro validation 0.484, sovrastima nel validation (0.518 prevista contro 0.484 osservata)
- PUNTI_OPERATIVI SHD: sens 90% → spec 0.446 [0.437-0.455], PPV 0.683 (soglia 0.296); spec 90% → sens 0.542 [0.532-0.554], PPV 0.878 (0.739); Youden 0.682/0.799 (0.586)
- PUNTI_OPERATIVI LVEF≤45: sens 90% → spec 0.573 [0.563-0.580], NPV 0.944 (soglia 0.073); spec 90% → sens 0.669 [0.655-0.684], PPV 0.695 (0.372); Youden 0.755/0.835 (0.228). PPV/NPV validi solo con prevalenza 0.57 (SHD) e 0.25 (LVEF) di pazienti con eco
- DERIVAZIONI_PIU_IMPORTANTI: canale 6 (toracico, non nominato) con calo SHD −0.026 [−0.035, −0.018] (fold 0) e −0.013 [−0.022, −0.003] (fold 1); tutte le altre < 0.01 (informazione ridondante)
- FINESTRE_PIU_IMPORTANTI: nessuna; ogni finestra da 0.5 s fa calare l'AUROC di ≤ 0.004 (SHD) e ≤ 0.006 (LVEF): la rete usa tutto il tracciato in modo diffuso
- COERENZA_FOLD0_FOLD1: solo il messaggio principale (canale 6 primo, altri piccoli); ordine fine non coerente (Spearman 0.20 / 0.26 per derivazioni, −0.02 / 0.36 per finestre, p ≥ 0.31)
RACCOMANDAZIONE:
- Q6/Q9: riportare che la rete è discretamente calibrata ma sovrastima in popolazioni con minore prevalenza; se si mostrano probabilità, ricalibrare (isotonica) e dichiarare che PPV/NPV dipendono dalla prevalenza. Nessun candidato nuovo da Q5.
- Interpretabilità: dichiarare «nessuna finestra temporale o derivazione sufficiente da sola a spiegare la predizione, a parte un canale toracico»; non fare affermazioni causali.
