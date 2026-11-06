# Report P8 · synthesis
- Agente: synthesizer
- Data: 2026-11-06 16:30
- Stato: OK

## Obiettivo
Riassumere per partner clinici e responsabili, senza gergo, i risultati delle fasi P0-P7: dove la predizione di cardiopatia strutturale (SHD) funziona, dove no, e quanto ci si può fidare. Promemoria: SHD NON coincide con scompenso cardiaco.

## Input letti
- `python3 tools/state.py brief` e Handoff di P0-P7, R1-P3, R1-P4, R1-P5.
- Report completi: `reports/P2_conditions.md`, `P4_waveform.md`, `P6_review.md`, `P7_final-test.md`; `reports/final_test_table.md`; `agentic/DECISIONS.md` (via brief); `agentic/REPORT_TEMPLATE.md`.
- `results/registry.csv` non riletto direttamente: i conteggi sono quelli del brief e di P6 (vedi Problemi aperti).
- Il test set non è stato aperto; i numeri di test vengono solo da `reports/P7_final-test.md`.

## Cosa è stato fatto
Solo sintesi di documenti già esistenti. Nessun calcolo nuovo, nessuno smoke test. Le versioni dei pacchetti sono state lette con `python3 -c`. Dove due fonti discordano, la discordanza è riportata (sezioni 3, 4, 5 e Problemi aperti).

## Risultati

### Documento di sintesi per i partner

### 1. In breve
- **Domanda.** Dall'ECG a 12 derivazioni si può prevedere una cardiopatia strutturale moderata o grave (SHD, composito di 11 condizioni all'ecocardiogramma)? Per quali condizioni e per quali gruppi di pazienti? Il QRS cambia la relazione con la LVEF (frazione di eiezione)?
- **Trovato.** Le sole 7 misure ECG tabulari (età, frequenza, PR, QRS, QTc…) danno una predizione moderata: AUROC 0.746 su validation. Usando la forma d'onda grezza con una rete neurale (CNN) sale a **0.830 [0.820-0.841] sul test finale** (classe: buona).
- **Trovato (LVEF).** Il modello tabulare per LVEF ≤ 45% ha AUROC test 0.793 [0.776-0.809] (classe: moderata).
- **NON trovato.** Nessun sottogruppo, regola semplice o cluster di pazienti in cui la predizione sia credibilmente migliore del modello globale (0 su 1950 regole e 4 cluster). L'ipotesi che il QRS modifichi la relazione con la LVEF resta inconclusiva.
- **Significato.** Le misure tabulari non bastano; la strada sono le forme d'onda. Non servono modelli separati per sottogruppo: un modello globale sulle onde è meglio.
- **Fiducia.** Il test è stato usato una sola volta e non mostra ottimismo (val→test −0.007 e −0.029, cioè il test è un po' migliore). Revisione indipendente: PASS. Cautele: un solo seed, centro unico, solo pazienti con eco, calibrazione non valutata, nessuna validazione esterna.
- Le etichette delle righe senza LVEF misurata sono tutte «negative» per costruzione: l'AUROC è gonfiata di circa 0.01.

### 2. Risposte per obiettivo
Classi (PLAYBOOK): buona = AUROC test ≥ 0.80 con limite inferiore IC ≥ 0.75; moderata 0.70-0.80; debole < 0.70. La classe si giudica solo sul test. Le condizioni singole non sono state portate al test: per esse i valori sono di validation e le classi sono indicative.

**(a) SHD composito: esito positivo, ma solo con le forme d'onda.**

| Modello | Split | AUROC [IC95%] | Classe |
|---|---|---|---|
| CNN su forma d'onda (`cnn_multilabel`) | test, n=5442, 2318 positivi | 0.830 [0.820-0.841] (AUPRC 0.802) | buona |
| CNN, solo righe con LVEF misurata | test, n=4827 | 0.817 [0.807-0.829] | buona |
| Modello tabulare (hist_gb, 7 feature) | test, stessi pazienti | 0.762 [0.750-0.774] | moderata |
| Modello tabulare | validation, n=4626 | 0.746 [0.733-0.759] | moderata |
| Solo età e sesso | validation | 0.659 [0.642-0.675] | debole |

Il vantaggio della CNN è +0.077 [0.063, 0.091] su validation (IC appaiato). Su test la differenza è +0.069, ma ottenuta da due IC separati: l'IC appaiato non è stato calcolato.

**(b) Singole condizioni** (validation, 1 ECG per paziente; miglior modello tabulare scelto sul validation, quindi leggermente ottimistico; CNN con una sola configurazione):

| Condizione | n pos val | Tabulare [IC95%] | CNN [IC95%] | Classe indicativa (CNN / tabulare) |
|---|---|---|---|---|
| Stenosi aortica | 252 | 0.844 [0.819-0.868] | 0.831 [0.807-0.851] | buona / buona (CNN = tabulare, diff. −0.012 [−0.039, 0.014]) |
| Rigurgito mitralico | 282 | 0.770 [0.741-0.793] | 0.839 [0.817-0.860] | buona / moderata |
| Rigurgito tricuspidale | 305 | 0.756 [0.730-0.783] | 0.843 [0.821-0.865] | buona / moderata |
| Disfunzione sistolica VD | 368 | 0.764 [0.737-0.787] | 0.885 [0.868-0.904] | buona / moderata |
| LVEF ≤ 45% | 866 | 0.764 [0.748-0.782] | 0.889 [0.876-0.902] | buona / moderata |
| PASP ≥ 45 | 581 | 0.688 [0.665-0.709] | 0.775 [0.755-0.794] | moderata / debole |
| Spessore parete VS ≥ 13 | 877 | 0.686 [0.669-0.705] | 0.751 [0.734-0.766] | moderata / debole |
| TR max ≥ 32 | 267 | 0.688 [0.656-0.720] | 0.785 [0.756-0.815] | moderata / debole |
| Versamento pericardico | 52 | 0.678 [0.604-0.750] | 0.754 [0.687-0.821] | non affidabile (pochi positivi) |
| Rigurgito aortico | 62 | 0.674 [0.592-0.736] | 0.768 [0.697-0.818] | non affidabile (pochi positivi) |
| Rigurgito polmonare | 21 | 0.613 [0.500-0.751] | 0.747 [0.602-0.870] | non valutabile (21 positivi) |

Nota: per condizioni rare l'AUROC alta non implica un buon valore predittivo (AUPRC tabulare stenosi aortica 0.265, prevalenza 5.4%). Per la stenosi aortica la discriminazione tabulare sembra guidata dall'età: 0.634 [0.542-0.734] a 65-74 anni contro 0.845 [0.820-0.876] sotto i 100 ms di QRS. Il tabulare del rigurgito aortico è 0.674 in P2 e 0.632 in P4 (modelli diversi scelti in fasi diverse).

**(c) LVEF e ruolo del QRS.**

| Misura | Split | Valore [IC95%] | Classe |
|---|---|---|---|
| LVEF ≤ 45%, modello tabulare (`lvef_lte45_global`) | test, 962 positivi | AUROC 0.793 [0.776-0.809], AUPRC 0.474 | moderata |
| Stessa, solo righe con LVEF misurata | test, n=4827 | 0.780 [0.763-0.797] | moderata |
| LVEF ≤ 45%, testa della CNN (non candidata) | validation | 0.889 [0.876-0.902] | buona (indicativa) |
| LVEF continua, hist_gb | validation, n=4112 | MAE 9.54 [9.31-9.79], R² 0.163 [0.136-0.191] (baseline MAE 11.19, R² −0.02) | debole come stima puntuale |

Il QRS modifica la relazione con la LVEF? **Inconclusivo**: tre test non concordano.
- (i) Modelli separati per fascia di QRS: ΔR² tra +0.004 e +0.009, nessun beneficio rilevante.
- (ii) Interazione feature×QRS: significativa come p-value (p < 1e-4 continuo), ma in cross-validation R² passa da 0.1401 a 0.1413: trascurabile.
- (iii) Coppie di pazienti: il QRS è più largo nei pazienti con LVEF bassa di circa 2-8 ms (58-72% delle coppie). È un'associazione. L'ampiezza dipende dalle variabili di appaiamento (si riduce a 2-4 ms appaiando PR e QTc) e in validation l'IC include 0 per tre appaiamenti su quattro.

### 3. Dove la predizione funziona
Tabella dei candidati ammessi dalla revisione P6 e valutati sul test (una sola volta).

| Candidato | Tipo | Feature | n test (pos) | AUROC val [IC] | AUROC test [IC] | Calo val→test | Globale stessi pazienti (test) |
|---|---|---|---|---|---|---|---|
| `cnn_multilabel` | modello globale, target SHD | forma d'onda (12 derivazioni) | 5442 (2318) | 0.823 [0.811-0.834] | 0.830 [0.820-0.841] | −0.007 (nessun calo) | 0.762 [0.750-0.774] (tabulare P1) |
| `lvef_lte45_global` | modello globale, target LVEF ≤ 45% | 7 feature ECG | 5442 (962) | 0.764 [0.748-0.782] | 0.793 [0.776-0.809] | −0.029 (nessun calo) | non pertinente (0.775, è un modello SHD) |

- **Sottogruppi: 0 candidati. Cluster: 0 candidati.** Non esiste una riga da riportare per «sottogruppo/cluster».
- Il «confronto col globale» dei candidati è quindi con il modello tabulare globale.
- Il calo val→test è negativo: il test è andato meglio. È compatibile col caso (prevalenza SHD 0.426 su test, 0.430 su validation) e non va letto come miglioramento reale; gli IC si sovrappongono.
- Equità della CNN (validation, 16 strati per etnia, setting, sesso, età): la CNN supera il tabulare in 16/16 stime puntuali. I valori più bassi sono procedurale 0.748 [0.674-0.818] (n=177) e ≥ 75 anni 0.788 [0.761-0.813]. Sesso: F 0.817 [0.799-0.835], M 0.822 [0.804-0.837]. Gli IC si sovrappongono. L'IC appaiato è > 0 in 14 o 15 strati su 16, a seconda del seed (asian: limite inferiore +0.007 con un seed, −0.001 con l'altro).

### 4. Cosa non ha funzionato e perché conta
- **Sottogruppi per regole semplici (P2):** 1950 regole provate sul train, 15 verificate su validation, 0 credibili. Esempio: la migliore sul composito (età < 52 e PR ≥ 174, n=187) ha 0.762 [0.694-0.837] contro 0.746 globale, margine +0.015 con IC inferiore sotto il globale. Esempio di crollo: per il rigurgito mitralico una regola con margine +0.041 in train è scesa a −0.097 in validation. Una regola con AUROC 0.950 per la stenosi aortica aveva solo 8 positivi: non valutabile.
- **Modelli locali per strato (P2):** su 19 strati del composito, nessuna differenza positiva significativa. Per stenosi aortica e rigurgito mitralico: 0 positive su 28 confronti, 6 significativamente negative. Conviene il modello globale.
- **Cluster di pazienti (P5):** k=4 (silhouette 0.569). Su 40 confronti appaiati, 2 con IC > 0 (stenosi aortica, cluster 2: +0.021 [+0.003, +0.038] e +0.018 [+0.002, +0.034]) e 1 con IC < 0. I due positivi non reggono: AUROC locale 0.812 [0.761-0.868] < globale 0.844, p empirico 0.77 (30 permutazioni). I cluster differiscono per prevalenza di SHD (chi-quadro 281.7), non per predicibilità. La stabilità (ARI 0.999) è dominata dal sesso binario (0.976 con tutte le feature); l'analisi senza sesso non è stata fatta.
- **Ipotesi QRS (P3):** inconclusiva (sezione 2c).
- **Singole condizioni rare:** versamento pericardico, rigurgito aortico e polmonare hanno 21-62 positivi in validation: IC troppo larghi per qualunque conclusione.
- **Perché conta:** non ci sono sottogruppi «facili» su cui il modello tabulare sia affidabile. Per ottenere accuratezza buona serve il segnale della forma d'onda.
- **Anomalia sulle etichette (scoperta in P6):** le righe senza `lvef_value` (train 6457, 8.9%; validation 514, 11.1%) hanno tutti i 12 flag = 0. Non è leakage di ingresso, ma può essere un artefatto. Potrebbero essere ecocardiogrammi incompleti codificati come negativi. Effetto sulla validation: CNN 0.823 → 0.812, tabulare 0.746 → 0.732, LVEF ≤ 45% 0.764 → 0.749. Sul test: CNN 0.830 → 0.817, LVEF ≤ 45% 0.793 → 0.780.

### 5. Quanto fidarsi
- **Configurazioni provate:** 511 nel registro del brief (22 esperimenti). Di queste, 433 hanno usato il validation (P6); le altre 78 sono clustering valutati solo sul train. In più 1950 regole di sottogruppo valutate solo sul train.
- **Selezione sul validation:** il miglior modello tabulare per condizione e il target dei cluster sono stati scelti guardando il validation, quindi quei valori sono leggermente ottimistici. La CNN ha una sola configurazione e l'epoca finale (12) è stata fissata a priori; l'AUROC monitorata ha un picco di 0.831 all'epoca 8 contro 0.823 alla 12.
- **Revisione indipendente (P6):** PASS al round 2. Le revisioni R1-P3, R1-P4 e R1-P5 sono state rifatte e riprodotte, con 0 discrepanze oltre 0.01 AUROC. Non c'è leakage di ingresso. Affermazioni: 41 supportate, 5 deboli, 0 non supportate. Le deboli includono: «IC > 0 in 15/16 strati» (dipende dal seed), ARI senza sesso non testato, AUROC LVEF ≤ 45% gonfiata dalle righe senza LVEF. Il totale 5 somma 3 deboli del round 2 e 2 del round 1 aggiornato; queste ultime sembrano coincidere con voci del round 2, quindi le distinte potrebbero essere 3.
- **Calo validation→test:** −0.007 (CNN) e −0.029 (LVEF ≤ 45%). Nessun calo superiore a 0.05: nessuna evidenza di ottimismo da selezione.
- **Fiducia complessiva:** alta per «la forma d'onda porta molto più segnale delle 7 feature» (confermato su test). Media per il valore assoluto dell'AUROC (un seed, un centro, etichette con anomalia). Alta per il risultato negativo sui sottogruppi, con la riserva che la ricerca (regole semplici su 7 feature, 4 cluster) non è esaustiva.

### 6. Limiti
- **SHD non è scompenso cardiaco.** È un'etichetta ecocardiografica; un AUROC alto per SHD non dice nulla sul rischio di scompenso.
- **Etichetta composita:** SHD aggrega 11 condizioni molto diverse; la predicibilità va da 0.84 (stenosi aortica, tabulare) a 0.67 (rigurgito aortico).
- **Centro unico, nessuna validazione esterna.** Non sappiamo se il modello regga in altri ospedali, apparecchi o popolazioni.
- **Solo pazienti con un'ecocardiografia entro un anno dall'ECG:** bias di selezione. Prevalenza SHD 43-52%, lontana da quella di uno screening. Valori predittivi in popolazione generale non stimabili.
- **Equità:** nei gruppi con pochi pazienti (procedurale n=177, asian n=134) gli IC sono larghi. La CNN è più debole sopra i 75 anni e nel setting procedurale; differenze tra gruppi non testate formalmente.
- **Calibrazione non valutata.** Su validation il punteggio medio della CNN è 0.455 contro prevalenza 0.430; hist_gb 0.503 (shift di prevalenza train 0.524 / validation 0.430). Nessuna soglia operativa è stata proposta.
- **Un solo seed** per la CNN: variabilità tra addestramenti ignota.
- **Etichette con LVEF mancante** tutte negative (sezione 4).
- **Associazione, non causalità,** per qualsiasi risultato sul QRS.

### 7. Glossario
- **AUROC:** probabilità che il modello dia un punteggio più alto a un paziente malato che a uno sano scelti a caso. 0.5 = caso, 1 = perfetto.
- **AUPRC:** simile all'AUROC, ma sensibile alla rarità della condizione; il valore di riferimento del caso è la prevalenza.
- **IC95%:** intervallo che, con il campione disponibile, contiene probabilmente il valore vero; più è stretto, più la stima è precisa.
- **Training / validation / test:** dati per addestrare il modello / per confrontare e scegliere / tenuti chiusi e aperti una sola volta per la stima finale.
- **Leakage:** il modello vede, direttamente o indirettamente, informazioni sull'esito (ad esempio misure dell'ecocardiogramma) e sembra più bravo di quanto sia.
- **Sottogruppo:** gruppo di pazienti definito da una regola semplice (es. età < 52 e PR ≥ 174 ms).
- **Cluster:** gruppo di pazienti simili trovato automaticamente dall'algoritmo, senza usare l'esito.
- **Permutazione:** si rimescolano le etichette per vedere quanto spesso il caso produce un risultato bello quanto quello osservato (p empirico).
- **CNN:** rete neurale che legge direttamente la forma d'onda dell'ECG.
- **Calibrazione:** quanto la probabilità dichiarata dal modello corrisponde alla frequenza reale della malattia.

### 8. Come riprodurre
- **Ambiente** (letto con `python3 -c`): Python 3.14.4, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, torch 2.15.0+cu132. GPU: RTX 5080 (P0).
- **Seed:** 42 ovunque. Il bootstrap di verifica di P6 usa seed 7 e il controllo del contratto seed 123.
- **Stato e pipeline:** `python3 tools/state.py brief`; driver `tools/run_pipeline.py`; test finale solo con `tools/final_eval.py` (già eseguito una volta: non rilanciare).
- **Script:** `experiments/P1/baseline.py`, `experiments/P2/conditions.py`, `experiments/P3/lvef_qrs.py`, `experiments/P4/train.py` e `model_def.py`, `experiments/P5/select_cluster.py` e `local_vs_global.py`, `experiments/P6/verify_round2.py`, `experiments/R1-P3/pairs_dedup.py`, `experiments/R1-P4/strat.py`.
- **Candidati:** `results/P4/candidates/cnn_multilabel`, `results/P3/candidates/lvef_lte45_global`, comparatore `results/P1/candidates/global_best`; congelati in `agentic/FROZEN.json`.
- **Risultati:** `results/final/test_results.json`, `reports/final_test_table.md`, `results/registry.csv`.
- I dati EchoNext 1.1.1 (PhysioNet) vanno messi nel `data_dir` indicato dal brief. Ogni script ha una modalità `debug_n=2000`.

### 9. Prossimi passi (per valore atteso)
1. **Chiarire le etichette senza LVEF** con i curatori del dataset o i clinici (8.9-11.1% delle righe, tutte negative): decide se rifare le stime escludendole.
2. **Validazione esterna** della CNN su un altro centro, con valutazione della calibrazione e scelta di una soglia operativa con i clinici.
3. **Più seed e intervalli di variabilità** della CNN; calcolare l'IC appaiato CNN vs tabulare sul test (solo con nuovi dati: il test attuale è già usato).
4. **Obiettivi singoli con la CNN** (LVEF ≤ 45%, disfunzione VD, rigurgiti): oggi sono alti solo sul validation e la testa LVEF≤45 non è stata candidata. Pre-specificarli e valutarli su dati nuovi.
5. **Ipotesi QRS e sottogruppi:** provarle sugli embedding della CNN e su un campione più ampio (con ARI senza il sesso), perché sulle sole 7 feature tabulari non c'è evidenza.

## Decisioni e motivazioni
- Il documento riporta solo numeri presenti nei report; i valori di test vengono da P7.
- Le classi delle singole condizioni sono indicative (solo validation): il test non le ha valutate.
- Dove le fonti discordano ho riportato entrambi i valori (rigurgito aortico tabulare 0.674 in P2 contro 0.632 in P4; IC appaiato 14/16 contro 15/16; 511 contro 433 configurazioni).

## Configurazioni provate
0 nuove configurazioni, nessun uso del validation né del test in P8. Totale ereditato: 511 (di cui 433 con validation), come da brief e P6.

## Artefatti
- `reports/P8_synthesis.md`, copia in `reports/FINAL_REPORT.md`.

## Problemi aperti
- Discordanza di conteggio: il brief dà 511 configurazioni, P6 dice 433 con validation (+78 clustering solo train). Si accordano (511 − 78 = 433), ma `results/registry.csv` non è stato riletto in questa fase.
- P6 conta 5 affermazioni deboli (3 round 2 + 2 round 1 aggiornato), probabilmente con sovrapposizione: le distinte potrebbero essere 3.
- La data di P6 (16:10) è successiva a quella di P7 (15:55) nei report: probabile incoerenza di timestamp, senza effetto sui numeri.
- IC appaiato CNN vs tabulare sul test non disponibile.
- Le classi delle singole condizioni (tabella 2b) sono basate sul validation.
- Calibrazione, seed multipli, validazione esterna e significato delle righe senza LVEF restano aperti.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- Documento finale in `reports/P8_synthesis.md` e `reports/FINAL_REPORT.md`; sezioni: 1 In breve, 2 Risposte per obiettivo, 3 Dove funziona, 4 Cosa no, 5 Fiducia, 6 Limiti, 7 Glossario, 8 Riproduzione, 9 Prossimi passi.
- SHD composito: CNN test 0.830 [0.820-0.841] (buona; 0.817 [0.807-0.829] su righe con LVEF); tabulare 0.762 [0.750-0.774].
- LVEF ≤ 45% tabulare: test 0.793 [0.776-0.809] (moderata); QRS-modifica inconclusiva.
- 0 sottogruppi e 0 cluster credibili (1950 regole, 40 confronti cluster, p=0.77).
- Calo val→test −0.007 e −0.029; revisione P6 PASS (41/5/0); 511 configurazioni (433 con validation).
- Limiti: SHD ≠ scompenso, centro unico, solo pazienti con eco, un seed, calibrazione non valutata, righe senza LVEF negative.
RACCOMANDAZIONE:
- P9: costruire il notebook dai numeri di questo report; non riaprire il test.
- Chiarire con i partner il significato delle righe senza LVEF prima di diffondere i numeri.
