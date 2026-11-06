# Report R1-P5 · review-clusters
- Agente: cluster-analyst
- Data: 2026-11-06 15:40
- Stato: OK

## Obiettivo
Rivedere P5 dopo la richiesta del revisore: (1) registrare con `log-exp` le 43 configurazioni su validation mancanti dal registro; (2) correggere il conteggio delle configurazioni; (3) dichiarare che i soli due confronti appaiati con IC95% > 0 (stenosi aortica, cluster 2) non sono confermati dalla permutazione; (4) precisare che l'ARI 0.999 è dominato dal sesso binario. Nessun nuovo calcolo sui modelli: si rileggono i risultati già salvati.

## Input letti
- Handoff P1 (AUROC val globale SHD 0.746 [0.733-0.759]) e P2 (nessun sottogruppo credibile); Handoff P5 e `reports/P5_clusters.md`.
- `results/P5/local_vs_global.csv` (53 righe: 3 globali, 10 cluster-target × (1 riferimento globale sugli stessi pazienti + 4 locali)), `results/P5/permutation.csv`, `results/P5/selection_summary.json` (valori già riportati in P5).
- `results/registry.csv` e `python3 tools/state.py brief`.

## Cosa è stato fatto
1. Verificato che il registro contava per P5 solo 125 esperimenti (78 selezione senza validation + 47 smoke test), non i 43 della run completa su validation.
2. Registrata la run completa: `log-exp --phase R1-P5 --uses-val --n-configs 43` (E0021). Il registro attribuisce quindi le 43 a «R1-P5», non a «P5»: la somma P5 + R1-P5 = 168 coincide col conteggio dichiarato.
3. Riletti i 40 confronti locale vs globale appaiati (10 coppie cluster×target × 4 modelli locali: logreg, Lasso, hist_gb, locale con selezione all'indietro) e contati quelli con IC95% della differenza tutto > 0 o tutto < 0.
4. Nessuno script nuovo: solo lettura di CSV aggregati. Nessun dato a livello di paziente, test set non toccato. Smoke test non applicabile (nessun modello riaddestrato).

## Risultati
Tutti i valori sono su validation (4626 pazienti, un ECG ciascuno); Δ = AUROC modello locale − AUROC del globale hist_gb sugli stessi pazienti, IC95% bootstrap appaiato.

**Confronti appaiati su 40**
| esito | n | dettaglio |
|---|---|---|
| IC95% Δ interamente > 0 | 2 | stenosi aortica, cluster 2 (563 pazienti, 73 positivi): logreg +0.021 [+0.003, +0.038]; Lasso +0.018 [+0.002, +0.034] |
| IC95% Δ interamente < 0 | 1 | SHD cluster 1, hist_gb −0.010 [−0.017, −0.004] |
| IC include 0 | 37 | |

Gli stessi due confronti positivi condividono cluster, target e famiglia di modelli lineari (non sono indipendenti). Il locale con selezione all'indietro nello stesso cluster dà +0.013 [−0.004, +0.029] (include 0); hist_gb locale −0.008 [−0.034, +0.017].
Su 40 confronti, con IC al 95%, ci si aspetta circa 1-2 IC positivi per solo caso (nominalmente 2.5% per coda, ma i confronti sono correlati): 2 su 40 è compatibile con il rumore.

**Permutazione (30 permutazioni delle etichette, procedura ripetuta per cluster)**: p empirico sul massimo guadagno locale−globale = 0.77 per la stenosi aortica (0.52 SHD, 0.94 rigurgito mitralico). Il minimo p ottenibile con 30 permutazioni è 1/31 = 0.032. Quindi il +0.021 non è confermato: il massimo guadagno osservato è comune sotto etichette mescolate. Inoltre, anche nel cluster 2 il modello locale (logreg 0.812 [0.761-0.868]) resta sotto l'AUROC globale su tutta la validation (0.844 [0.819-0.868]): non è un sottogruppo credibile (PLAYBOOK: serve Δ ≥ 0.02 con limite inferiore sopra il globale complessivo; qui 0.761 < 0.844).

**Stabilità ARI 0.999**: il clustering finale usa 3 feature (sex, atrial_rate, qrs_duration). Il sesso è binario e separa nettamente due gruppi (cluster 0 e 1: QRS ~84 e ~91 ms, differiscono essenzialmente per sesso), quindi l'accordo tra le due metà è quasi banale. L'ARI 0.999 misura la riproducibilità di una partizione guidata da una variabile a due valori più due code (QRS largo, frequenza atriale alta), non una struttura multidimensionale scoperta. Con tutte le feature l'ARI è 0.976 (silhouette 0.205), contro 0.569 con 3 feature. La silhouette alta è in parte effetto della dimensionalità ridotta. Questa è una lettura interpretativa dai profili salvati, non un nuovo esperimento di sensibilità (non condotto: ad esempio ARI senza sesso).

**Conferma conclusioni P5**: CLUSTER_CREDIBILI = [] invariato; nessun candidato da congelare. SHD ≠ scompenso cardiaco.

## Decisioni e motivazioni
- Registrare le 43 config come fase R1-P5 (il registro è gestito solo da `tools/state.py` e non permette di riassegnarle a P5). Il totale corretto per P5 è 168.
- Dichiarare i due Δ positivi come esplorativi, non confermati (p=0.77), non candidati.
- Non rieseguire permutazioni più numerose: il verdetto negativo non cambia e il costo non è giustificato.

## Configurazioni provate
- Correzione del conteggio P5: **168 configurazioni** = 78 selezione feature (solo train, senza validation, E0014) + 47 smoke test (E0015, `--uses-val`) + 43 run completa su validation.
- Registro: P5 = 125 righe-config; R1-P5 = 43 (E0021, `--uses-val`, retroattivo); totale 168. Config che hanno usato la validation: 47 + 43 = 90.
- In questa fase nessuna nuova configurazione è stata provata (solo registrazione e lettura dei risultati). Totale pipeline dopo la registrazione: 510 (da `brief`).
- La validation non è stata usata per scegliere k né le feature di clustering. Il numero di confronti appaiati (40) va tenuto presente per la multiplicità.

## Artefatti
- `results/P5/local_vs_global.csv`, `permutation.csv`, `selection_summary.json` (invariati).
- `results/registry.csv` (riga E0021).
- Nessun candidato (`results/P5/candidates/` non esiste).

## Problemi aperti
- Il clustering su embedding CNN (P4) non è stato eseguito (passo facoltativo).
- Solo 30 permutazioni: p minimo 0.032, risoluzione limitata.
- Nessuna analisi di sensibilità del clustering senza il sesso (non richiesta): resta non verificato quanto la struttura regga senza di esso.
- Cluster 3 (135 pazienti in validation): IC molto larghi.
- Le 43 config compaiono nel registro con fase R1-P5 invece di P5.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- N_CONFIGURAZIONI: 168 (78 senza validation + 47 smoke + 43 run completa su validation; tutte ora registrate: P5 125 + R1-P5 43, E0021).
- CONFRONTI_APPAIATI: 40; IC95% Δ > 0 in 2 (stenosi aortica cluster 2: logreg +0.021 [+0.003, +0.038], Lasso +0.018 [+0.002, +0.034]); IC < 0 in 1; IC con 0 in 37.
- P_EMPIRICO: 0.77 sul massimo guadagno per la stenosi aortica (0.52 SHD, 0.94 mitrale; 30 permutazioni): i due +Δ non sono confermati.
- CLUSTER_CREDIBILI: [] (cluster 2 locale 0.812 [0.761-0.868] < globale 0.844 [0.819-0.868]).
- STABILITA_ARI: 0.999 con 3 feature, dominato dal sesso binario (0.976 con tutte le feature).
RACCOMANDAZIONE:
- P6: nessun candidato da P5; riportare il risultato negativo e citare 40 confronti e 2 IC positivi non confermati.
- Usare 168 come conteggio delle configurazioni P5 (125 + 43 registrate come R1-P5).
