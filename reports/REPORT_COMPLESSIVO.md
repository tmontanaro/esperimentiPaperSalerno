# Report complessivo EchoNext: ECG e cardiopatia strutturale
Data: 2026-11-09. Copia autonoma del report Q9 (pipeline 1 e 2). Solo numeri aggregati; il test è chiuso.

Indice: In breve · A. Obiettivo · B. Risultati · C. Prove fatte · D. Fattibile e non · E. Cosa si può ancora fare · F. Quanto fidarsi · Glossario · Come riprodurre

## In breve
- Domanda: dall'elettrocardiogramma (ECG) a 12 derivazioni si può prevedere una cardiopatia strutturale (SHD) di grado moderato o grave, o singole condizioni, o una frazione di eiezione (LVEF) bassa? E il QRS (la durata dell'impulso che attiva i ventricoli) cambia il legame tra ECG e LVEF?
- Risposta sulle misure tabulari: le sole 7 misure ECG già calcolate dall'apparecchio (età, frequenza, PR, QRS, QTc…) danno una predizione moderata: AUROC 0.746 [0.733-0.759] su validation per SHD. L'AUROC è un voto da 0.5 (caso) a 1 (perfetto).
- Risposta sulla forma d'onda: una rete neurale (CNN) che legge il tracciato grezzo fa molto meglio. Sul test, nella popolazione principale, SHD 0.824 [0.813-0.836], LVEF ≤ 45% 0.897 [0.886-0.908], disfunzione del ventricolo destro 0.889 [0.873-0.904], stenosi aortica 0.832 [0.811-0.857]. Tutti e 5 i candidati congelati sono confermati con i criteri scritti prima.
- Sottogruppi e cluster: nessun gruppo di pazienti in cui la predizione sia credibilmente migliore del modello globale (0 su 1950 regole e 4 cluster nella pipeline 1; 0 su 15 cluster nella pipeline 2). Questo è un risultato negativo solido.
- Ipotesi QRS: non supportata sopra la CNN (differenza di AUROC +0.0001 [-0.0005, +0.0006]). Sulle sole misure tabulari era rimasta inconclusiva.
- Quanto fidarsi: media-alta per «la forma d'onda porta più informazione delle 7 misure». Prudenza sul valore assoluto: un solo centro, solo pazienti con ecocardiogramma, nessuna validazione esterna, test letto due volte.
- Fattibile? VERDETTO_FATTIBILITA: IN PARTE. Sì come strumento di ricerca per classificare SHD, LVEF ≤ 45%, disfunzione VD e stenosi aortica con la CNN. No come strumento per scegliere sottogruppi «facili», né per stimare la LVEF come numero preciso, né per una decisione clinica senza validazione esterna e ricalibrazione.
- SHD non coincide con lo scompenso cardiaco: è un'etichetta dell'ecocardiogramma.

## A. Qual era l'obiettivo degli esperimenti
L'ecocardiogramma (eco) è l'esame che mostra il cuore, ma costa tempo e personale. L'ECG è rapido ed economico. Se dall'ECG si potesse capire chi ha probabilmente una cardiopatia strutturale, si potrebbe decidere meglio a chi fare l'eco. Il dataset è EchoNext (PhysioNet 1.1.1): ECG a 12 derivazioni con le etichette ricavate dall'eco.

Le tre domande:
1. SHD composito (`shd_moderate_or_greater_flag`): è l'unione di 11 condizioni dell'eco di grado almeno moderato (per esempio stenosi aortica, rigurgiti valvolari, ipertrofia). Si può prevedere, e per quali gruppi di pazienti o di misure ECG funziona meglio?
2. Singole condizioni (le colonne `*_flag`): quali sono prevedibili e quali no?
3. LVEF e QRS: la LVEF è la percentuale di sangue che il ventricolo sinistro spinge fuori a ogni battito; «LVEF ≤ 45%» è un valore basso. I medici ipotizzano che la durata del QRS modifichi il legame tra le misure ECG e la LVEF.

Promemoria: SHD non è lo scompenso cardiaco. Un buon punteggio per SHD non dice nulla sul rischio di scompenso.

## B. Risultati ottenuti
Come leggere le classi di accuratezza (decise prima): buona = AUROC ≥ 0.80 con limite inferiore dell'intervallo di confidenza (IC95%, il margine di incertezza) ≥ 0.75; moderata 0.70-0.80; debole sotto 0.70. La classe è definitiva solo sul test. Il test è stato letto due volte: nella pipeline 1 (P7) e nella pipeline 2 (Q8).

Popolazione principale «clean»: le righe con LVEF misurata (4827 nel test). Le righe senza LVEF hanno tutte le 12 etichette a zero, il che è anomalo; riportiamo anche «tutte le righe» (5442) come controllo.

### B1. SHD composito
| Modello | Dove | AUROC [IC95%] | Classe |
|---|---|---|---|
| Solo età e sesso | validation (pipeline 1) | 0.659 [0.642-0.675] | debole |
| 7 misure ECG, hist_gb | validation | 0.746 [0.733-0.759] | moderata |
| 7 misure ECG | test, tutte le righe | 0.762 [0.750-0.774] | moderata |
| 7 misure ECG | CV, clean (pipeline 2) | 0.712 [0.705-0.721] | moderata |
| CNN, 12 derivazioni (C1) | CV, clean | 0.815 [0.809-0.822] | buona |
| CNN, 12 derivazioni (C1) | test, clean | 0.824 [0.813-0.836] | buona |
| CNN, 12 derivazioni (C1) | test, tutte le righe | 0.837 [0.827-0.846] | buona |
| CNN, 3 derivazioni (C5) | test, clean | 0.818 [0.808-0.829] | buona |

La CNN batte il tabulare di +0.103 [0.096, 0.109] in CV (clean) e di +0.075 [+0.063, +0.086] sul test (tutte le righe). Con sole 3 derivazioni la perdita è piccola: -0.006 [-0.009, -0.003] in CV, 0.006 sul test.

### B2. Singole condizioni
Pipeline 2, CV clean, CNN. Buone: LVEF ≤ 45% 0.869, disfunzione sistolica del ventricolo destro (VD) 0.844, stenosi aortica 0.811. Moderate: spessore di parete ≥ 13 mm, rigurgito mitralico, tricuspidale e polmonare, versamento pericardico, pressione polmonare (PASP) ≥ 45, TR max ≥ 32. Debole: rigurgito aortico 0.692 [0.652-0.732]. Per la stenosi aortica il modello tabulare era migliore della CNN in CV. Sul test sono state valutate solo 3 condizioni (LVEF ≤ 45%, VD, stenosi aortica), oltre a SHD.

| Candidato | Test clean AUROC [IC95%] | Test tutte le righe | CV clean | Calo CV-test | Classe |
|---|---|---|---|---|---|
| C1 SHD, 12 deriv. | 0.824 [0.813-0.836] | 0.837 | 0.815 | -0.009 | buona |
| C2 LVEF ≤ 45%, 12 | 0.897 [0.886-0.908] | 0.905 | 0.869 | -0.028 | buona |
| C3 disfunzione VD, 12 | 0.889 [0.873-0.904] | 0.899 | 0.844 | -0.045 | buona |
| C4 stenosi aortica, 12 | 0.832 [0.811-0.857] | 0.841 | 0.811 | -0.021 | buona |
| C5 SHD, 3 deriv. | 0.818 [0.808-0.829] | 0.830 | 0.802 | -0.016 | buona |

Calo negativo vuol dire test sopra la CV: non lo leggiamo come un vero miglioramento (la prevalenza di SHD nel test è diversa e il modello finale usa più dati). Nella pipeline 1 le condizioni singole erano state valutate solo su validation (esempio: stenosi aortica tabulare 0.844 [0.819-0.868] contro CNN 0.831 [0.807-0.851]; rigurgito polmonare, 21 positivi, non valutabile).

### B3. LVEF e ruolo del QRS
- LVEF ≤ 45%: tabulare, test 0.793 [0.776-0.809] (moderata); CNN C2, test clean 0.897 [0.886-0.908] (buona), vantaggio sul tabulare +0.112 [+0.099, +0.127] (tutte le righe).
- LVEF come numero: in pipeline 1 il tabulare dava errore medio (MAE) 9.54 [9.31-9.79] punti e R² 0.163 (debole). In pipeline 2 la CNN dà MAE 7.48 e R² 0.482 contro tabulare MAE 10.45 e R² 0.156 (CV, clean): meglio, ma ancora un errore di circa 7 punti.
- QRS modifica la relazione? Pipeline 1: inconclusivo (i tre metodi non concordano; il QRS è più largo di circa 2-8 ms nei pazienti con LVEF bassa, è un'associazione). Pipeline 2, con la CNN: non supportata. Aggiungere informazione sul QRS migliora l'AUROC di +0.0001 [-0.0005, +0.0006] (p dopo correzione 0.79) e R² di +0.0004 [-0.0002, +0.0010]. Per fascia di QRS, LVEF ≤ 45%: <100 ms 0.840, 100-119 ms 0.867, ≥ 120 ms 0.883. Cautela: la CNN vede già il QRS nel tracciato, quindi non può «mostrare» un effetto che ha già assorbito.

### B4. Sottogruppi e cluster
- Pipeline 1, regole semplici: 1950 provate, 15 verificate su validation, 0 credibili. Cluster (4): 2 confronti positivi su 40 che non reggono (p empirico 0.77).
- Pipeline 2, sugli embedding della CNN (la rappresentazione interna del tracciato): 0 cluster credibili su 15; 3 gruppi di rischio (prevalenza 0.3, 0.65, 0.92) che sono solo fasce di punteggio. Nel rumore (permutazioni) il massimo guadagno atteso è +0.005 in media.
- Dove la CNN va un po' peggio (diagnostico, non causale): SHD con QRS < 100 ms 0.787 [0.778-0.795], -0.052; PR basso 0.771 [0.760-0.782], -0.054; QTc alto va meglio (0.821). Anche nel peggiore strato l'AUROC resta ≥ 0.77.

### B5. Calibrazione e punti operativi
Calibrazione = quanto la probabilità dichiarata corrisponde alla frequenza reale. In CV l'errore di calibrazione (ECE) è 0.031 [0.026-0.036] per SHD, ridotto a 0.003 con ricalibrazione isotonica; sul test, con probabilità grezze, è 0.056 (C1) e 0.060 (C5). Il modello sovrastima dove la prevalenza è più bassa. Soglie (CV): per sensibilità 90% la specificità di SHD è 0.446 [0.437-0.455]; per specificità 90% la sensibilità è 0.542 [0.532-0.554]. Per LVEF ≤ 45%: sensibilità 90% con specificità 0.573 [0.563-0.580] (valore predittivo negativo 0.944). I valori predittivi dipendono dalla prevalenza di questi pazienti (57% SHD, 25% LVEF ≤ 45%), che è quella di chi ha già fatto un'eco.

Interpretabilità: nessuna finestra di 0.5 s fa calare l'AUROC di più di 0.004 (SHD): la rete usa tutto il tracciato. Un solo canale toracico (non nominato) pesa un po' di più (calo -0.026 [-0.035, -0.018] in un fold, -0.013 in un altro); l'ordine fine tra le derivazioni non è stabile.

### B6. Tabella riassuntiva
| Obiettivo | Esito | Fiducia |
|---|---|---|
| SHD dalle 7 misure | moderato, 0.746 val | media |
| SHD dalla forma d'onda (CNN) | buono, test 0.824 clean | media-alta |
| LVEF ≤ 45% (CNN) | buono, test 0.897 clean | media-alta |
| Disfunzione VD (CNN) | buono, test 0.889 clean | media (manca il confronto tabulare) |
| Stenosi aortica (CNN) | buono, test 0.832 clean; non migliore del tabulare in CV | media-bassa (286 positivi) |
| Altre condizioni | moderate o deboli, solo CV | media |
| 3 derivazioni bastano | in parte: perdita 0.006 | media |
| LVEF come numero | MAE 7.48, R² 0.482 | media |
| QRS modifica la relazione | non supportata | media (la CNN vede il QRS) |
| Sottogruppi/cluster «facili» | nessuno | alta, ricerca non esaustiva |

## C. Prove fatte, passo passo

### Pipeline 1 (P0-P9)
P0, controllo dei dati. Volevamo capire se i dati erano utilizzabili e senza ripetizioni di pazienti tra gruppi. Abbiamo controllato file, etichette e forme d'onda. Risultato: 72475 ECG di addestramento (26218 pazienti), 4626 di validation; nessun paziente in comune tra train, validation e test. Abbiamo deciso di dividere sempre per paziente nelle verifiche.

P1, modello di partenza. Volevamo sapere quanto dicono le 7 misure ECG. Abbiamo provato 10 modelli (usando la validation, l'insieme che serve a scegliere). Il migliore, hist_gb (alberi di decisione), ha AUROC 0.746 [0.733-0.759]; con solo età e sesso 0.659. Conclusione: c'è segnale tabulare moderato.

P2, singole condizioni e sottogruppi. Volevamo sapere se qualche gruppo semplice (per esempio età < 52 e PR ≥ 174) sia prevedibile meglio. Abbiamo provato 1950 regole sull'addestramento e verificato le migliori 15 sulla validation. Nessuna credibile: la migliore ha 0.762 [0.694-0.837] contro 0.746 globale. I modelli per strato non battono quello globale.

P3, LVEF e QRS. Volevamo capire se il QRS cambia il legame con la LVEF. Abbiamo provato modelli per fascia, interazioni e coppie di pazienti simili. Esito: inconclusivo, effetti trascurabili (R² da 0.1401 a 0.1413).

P4, forma d'onda. Volevamo vedere se il tracciato grezzo contiene più informazione. Abbiamo addestrato una CNN (rete neurale per segnali). Risultato in validation: SHD 0.823 [0.811-0.834]. Decisione: usarla come candidato.

P5, cluster. Volevamo vedere se gruppi di pazienti simili hanno predicibilità diversa. Abbiamo formato 4 cluster e confrontato. Esito: i cluster differiscono per prevalenza, non per predicibilità; nessuno credibile.

P6, revisione indipendente. Un revisore ha rifatto i calcoli. Esito: PASS, 41 affermazioni supportate, 5 deboli, 0 non supportate. Scoperta: le righe senza LVEF hanno tutte le etichette a zero.

P7, primo test. Volevamo una stima finale onesta. Abbiamo aperto il test una volta su 2 candidati: CNN SHD 0.830 [0.820-0.841]; tabulare LVEF ≤ 45% 0.793 [0.776-0.809]. Nessun calo rispetto alla validation.

P8, sintesi; P9, notebook. Abbiamo scritto il documento per i partner e un notebook che ripete le analisi.

### Pipeline 2 (Q0-Q8)
Dopo la pipeline 1 restavano dubbi: la CNN era stata provata con un solo seed (un seed è un punto di partenza casuale dell'addestramento); la calibrazione non era valutata; le righe senza LVEF erano un problema. Si lavora con gli stessi dati, senza dati nuovi. Ora si usa la validazione incrociata (CV): si divide il materiale in 5 parti per paziente e si addestra 5 volte, a turno testando su una parte diversa.

Q0, inventario. Abbiamo verificato che i dati e i pesi della pipeline 1 fossero presenti e che l'ordine delle derivazioni fosse leggibile. Risultato: 77101 ECG, 30844 pazienti nello sviluppo; 91% (70130 ECG) in popolazione clean. Ordine dei canali verificato per quelli degli arti, non per i toracici.

Q1, CNN in CV. Volevamo una stima più solida con più seed. Abbiamo addestrato la CNN in 5 fold e 3 seed. Risultato: SHD 0.815 [0.809-0.822], variazione tra seed < 0.001; tabulare 0.712. Le righe senza LVEF cambiano il risultato di poco (+0.0020).

Q2, quali derivazioni servono. Abbiamo tolto derivazioni a gruppi. Un set di 3 derivazioni perde solo 0.006 su SHD e 0.008 su LVEF ≤ 45%; una sola derivazione perde 0.024-0.044. Decisione: C5 con 3 derivazioni. Dichiarato: «sufficiente» vuol dire perdita ≤ 0.01 con un seed.

Q3, LVEF e QRS con la CNN. Abbiamo confrontato la CNN con e senza informazione sul QRS (8 confronti, correzione di Holm: un aggiustamento per evitare falsi positivi quando si fanno molti confronti). Risultato: nessun effetto, +0.0001 [-0.0005, +0.0006].

Q4, cluster e sottogruppi sugli embedding. Abbiamo cercato gruppi di pazienti nello spazio interno della CNN. 0 credibili su 15; 137 confronti in tutto. Decisione: nessun candidato nuovo.

Q5, calibrazione e interpretabilità. Volevamo sapere se le probabilità sono affidabili e cosa guarda la rete. Risultato: ECE 0.031 grezza, 0.003 dopo ricalibrazione; la rete usa tutto il tracciato.

Q6, candidati. Prima di aprire il test abbiamo scritto 5 candidati e criteri di conferma: AUROC con IC inferiore ≥ 0.75 e calo da CV ≤ 0.05. I file sono protetti da impronte sha256 (un sigillo che rivela modifiche).

Q7, revisione. Verdetto PASS: 216 AUROC riprodotte (differenza massima 0.000), nessun paziente condiviso tra addestramento e verifica, 5 candidati ammessi; affermazioni 14 supportate, 5 deboli, 0 non supportate. Deboli: «3 derivazioni sufficienti», «CNN peggiore in alcuni strati», regole sugli errori.

Q8, secondo test. Con approvazione umana, test letto una volta sui 5 candidati: tutti confermati (tabella in B2). Il primo tentativo era fallito per un pacchetto mancante (torch) senza produrre metriche.

## D. Cosa è fattibile e cosa no
Criteri: classe di accuratezza sul test e conferma con regole scritte prima.

Fattibile (in parte): classificare con la CNN SHD (0.824), LVEF ≤ 45% (0.897), disfunzione VD (0.889) e stenosi aortica (0.832); farlo con 3 derivazioni (0.818); usare il tracciato intero al posto delle 7 misure (+0.075 per SHD, +0.112 per LVEF ≤ 45%).

Non fattibile con questi dati: trovare sottogruppi o cluster in cui sia credibilmente più facile; mostrare che il QRS modifichi la relazione; stimare la LVEF come numero preciso (MAE 7.48); confermare le condizioni rare (versamento pericardico 52 positivi, rigurgito aortico 62, polmonare 21 in validation); dire che la CNN batta il tabulare sulla stenosi aortica; usare le probabilità senza ricalibrazione; dichiarare lo scompenso.

## E. Cosa si può ancora fare
Con gli stessi dati: ricalibrare le probabilità per sottopopolazione; calcolare un confronto tabulare per VD e stenosi aortica in CV; analizzare con più seed le derivazioni; verificare con i curatori il significato delle righe senza LVEF; studiare con altri metodi il «peggio» a QRS stretto. Con dati nuovi: validazione esterna in altri ospedali, popolazione di screening, test non ancora letto, outcome (ricoveri, mortalità).

## F. Quanto fidarsi
- Configurazioni provate: 725 nel registro (esperimenti 40): pipeline 1 = 511, pipeline 2 = 214. Tutte quelle con la CV usano anche la validation, quindi i valori scelti guardando la validation sono leggermente ottimistici.
- Revisioni indipendenti: P6 (PASS) e Q7 (PASS).
- Test letto due volte (P7, Q8); la seconda lettura è meno indipendente perché chi ha progettato la pipeline 2 conosceva P7. Il test non va più usato.
- La popolazione del criterio di conferma (clean) non era nella preregistrazione ma è stata fissata prima di aprire il test. Il riferimento CV di C5 è a seed singolo.
- Il test risulta sopra la CV in tutti i candidati: non lo leggiamo come miglioramento. ECE sul test (0.056-0.060 per SHD) peggiore della CV.
- Limiti: un solo centro, solo pazienti con un'eco entro un anno (prevalenza SHD 43-57%), righe senza LVEF anomale, nessuna validazione esterna, calibrazione solo grezza sul test, il test non valuta sottogruppi.
- Discordanze tra fonti: la CNN SHD test 0.830 (P7) e 0.837 (C1, Q8) su tutte le righe sono modelli diversi (IC sovrapposti); il tabulare SHD vale 0.746 (val), 0.762 (test) e 0.712 (CV clean) perché sono popolazioni diverse; il tabulare per rigurgito aortico 0.674 in P2 e 0.632 in P4; l'occlusione del canale toracico -0.026 (Q5) contro -0.021 (Q7, campione diverso).

## Glossario
- AUROC: probabilità che un malato abbia punteggio più alto di un sano; 0.5 = caso, 1 = perfetto.
- AUPRC: simile, ma sensibile alla rarità della condizione.
- IC95%: intervallo in cui cade probabilmente il valore vero.
- CNN: rete neurale che legge il tracciato ECG grezzo.
- Cross-validation (CV): addestrare e testare a turno su parti diverse dei dati, separando per paziente.
- Validation / test: insieme per scegliere i modelli / insieme chiuso per la stima finale.
- Embedding: rappresentazione numerica interna con cui la rete riassume un ECG.
- Cluster: gruppo di pazienti simili trovato senza usare l'esito.
- Calibrazione (ECE): accordo tra probabilità dichiarata e frequenza reale; ECE basso è meglio.
- Leakage: il modello vede informazioni sull'esito e sembra più bravo.
- Holm: correzione per confronti multipli.

## Come riprodurre
Ambiente: Python con scikit-learn, torch e GPU (lettura delle versioni nella pipeline 1: P8). Seed 42. Stato: `python3 tools/state2.py brief`. CNN in CV: `python3 tools/cnn_cv.py run --out results/Qn/nome ... --time-budget-min 7`. Test: solo `tools/final_eval.py` (P7) e `tools/final_eval2.py` (Q8), già eseguiti: non rilanciare. Risultati: `results/Q1/main`, `results/final2/test_results.json`, `reports/final_test_table.md`, `reports/final_test2_table.md`, `results/registry.csv`. Notebook: pipeline 1 in `reports/P9_notebook/`, pipeline 2 in Q10.
