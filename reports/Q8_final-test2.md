# Report Q8 · final-test2
- Agente: final-evaluator2
- Data: 2026-11-09 17:30
- Stato: OK

## Obiettivo
Leggere e interpretare la seconda lettura del test set (la prima era P7) su 5 candidati CNN congelati in Q6, applicando i criteri di conferma scritti prima del test. Serve a dire con onestà quali risultati della cross-validation (CV) reggono su pazienti mai visti. Ricordiamo che SHD (cardiopatia strutturale) NON coincide con scompenso cardiaco.

## Input letti
- Handoff Q7 (verdetto PASS, 5/5 candidati ammessi, sha256 30/30 integre) e Handoff Q6 (candidati, criteri, riferimenti CV).
- `results/Q6/preregistration.md` (criteri) e regola 4 di `agentic/PLAYBOOK2.md` (popolazione clean).
- `results/final2/test_results.json` e `reports/final_test2_table.md`, prodotti da `tools/final_eval2.py`.
- Riferimenti CV clean (seed 0, da Q6): C1 0.815, C2 0.869, C3 0.844, C4 0.811, C5 0.802.

## Cosa è stato fatto
1. Approvazione umana registrata (2026-11-09 16:56). `tools/final_eval2.py` è stato eseguito una sola volta con esito positivo, su richiesta esplicita dell'utente. Questa fase NON l'ha rilanciato e non ha caricato il test.
2. Tentativo precedente fallito (da dichiarare): una prima esecuzione con il Python di sistema è terminata con `ModuleNotFoundError` su torch alla prima predizione di C1. Il test era stato caricato in memoria ma non è stata calcolata né stampata alcuna metrica e non è stato scritto alcun file. L'esecuzione è stata rilanciata con il `.venv` del progetto, dove torch è presente.
3. Ho letto i risultati aggregati e applicato i criteri senza modificare soglie né candidati. Nessuno script nuovo, nessuna configurazione provata.

Natura della lettura: è la **seconda lettura del test** (prima: P7). Chi ha progettato la pipeline 2 conosceva i risultati di P7, quindi i candidati non sono scelti in totale ignoranza del test. Candidati valutati: 5 (massimo ammesso 6).

Criterio «confermato» (preregistrazione + regola 4): AUROC test sulle righe con LVEF noto (clean, n=4827), IC95% inferiore ≥ 0.75 e calo rispetto all'AUROC CV clean ≤ 0.05. **La popolazione su cui applicare il criterio non era scritta nella preregistrazione**: è stata fissata con la regola 4 (clean) prima di aprire il test, come raccomandato da Q7. Tutte le righe (n=5442) sono riportate come sensibilità. Classe: buona = AUROC ≥ 0.80 con IC inf ≥ 0.75; moderata 0.70-0.80; debole < 0.70. Calo = AUROC CV meno AUROC test clean (negativo = test migliore della CV).

## Risultati
Tutti i valori sono su TEST (split chiuso), IC95% bootstrap per paziente. Colonne «tutte le righe» n=5442; clean n=4827. Positivi: SHD 2318, LVEF≤45% 962, disfunzione VD 419, stenosi aortica 286.

| Candidato (target) | Test clean AUROC [IC] | Test tutte le righe AUROC [IC] | CV clean | Calo CV−test | Classe | Confermato |
|---|---|---|---|---|---|---|
| C1 SHD, 12 deriv. | 0.824 [0.813-0.836] | 0.837 [0.827-0.846] | 0.815 | −0.009 | buona | sì |
| C2 LVEF≤45%, 12 | 0.897 [0.886-0.908] | 0.905 [0.893-0.915] | 0.869 | −0.028 | buona | sì |
| C3 disfunzione VD, 12 | 0.889 [0.873-0.904] | 0.899 [0.885-0.914] | 0.844 | −0.045 | buona | sì |
| C4 stenosi aortica, 12 | 0.832 [0.811-0.857] | 0.841 [0.817-0.863] | 0.811 | −0.021 | buona | sì |
| C5 SHD, 3 deriv. | 0.818 [0.808-0.829] | 0.830 [0.820-0.840] | 0.802 | −0.016 | buona | sì |

Calibrazione e metriche su tutte le righe (probabilità grezze, senza ricalibrazione; il file non riporta ECE/Brier sulle sole clean):

| Candidato | ECE | Brier | AUPRC |
|---|---|---|---|
| C1 | 0.056 | 0.163 | 0.813 |
| C2 | 0.025 | 0.081 | 0.751 |
| C3 | 0.017 | 0.051 | 0.499 |
| C4 | 0.008 | 0.044 | 0.266 |
| C5 | 0.060 | 0.167 | 0.806 |

Confronto con il tabulare (modello classico sulle 7 feature, da P1/P3). ATTENZIONE: il tabulare nel file è calcolato su **tutte le righe in scope**, non sulle sole clean, quindi va confrontato con la colonna «tutte le righe» e non con quella clean.

| Candidato | CNN (tutte le righe) | Tabulare (tutte le righe) | Differenza [IC95%] |
|---|---|---|---|
| C1 SHD | 0.837 | 0.762 [0.750-0.774] | +0.075 [+0.063, +0.086] |
| C2 LVEF≤45% | 0.905 | 0.793 [0.776-0.809] | +0.112 [+0.099, +0.127] |
| C5 SHD 3 deriv. | 0.830 | 0.762 [0.750-0.774] | +0.068 [+0.057, +0.081] |
| C3, C4 | 0.899 / 0.841 | non disponibile | non disponibile |

C3 e C4 non hanno un comparatore tabulare nel test. In CV C4 era peggiore del tabulare, quindi non si può dire che lo batta.

Confronto con P7 (primo test): CNN SHD 0.830 e tabulare LVEF≤45% 0.793. La CNN SHD attuale (C1, tutte le righe) è 0.837 [0.827-0.846], coerente con 0.830 di P7 (IC sovrapposti; sono modelli diversi, rifatti con il final-fit di Q6, e non è un test di differenza). Il tabulare LVEF≤45% è identico (0.793); C2 lo supera di +0.112.

Interpretazione, senza entusiasmo:
- Tutti e 5 i candidati soddisfano i criteri: IC inferiore minimo 0.808 sulle clean (soglia 0.75); calo massimo −0.045, cioè nessun calo (il test è sopra la CV).
- L'AUROC del test è **superiore** alla CV per tutti. Non lo leggiamo come un miglioramento reale. Cause possibili: shift di popolazione e prevalenza (prevalenza SHD nel test 0.426 su tutte le righe e 0.480 sulle clean, contro 0.575 nel train e 0.484 nel validation); il modello congelato è un ensemble di più seed addestrato su più dati, mentre il riferimento CV è un seed singolo su 4/5 dei dati; variabilità normale. Sul test singolo non è attribuibile.
- Le AUROC su tutte le righe sono 0.008-0.013 più alte delle clean. Le righe senza LVEF sono anomale; la popolazione clean è quella principale.
- Il riferimento CV di C5 è un seed singolo (criterio più facile da superare); il calo −0.016 è comunque piccolo. Le 3 derivazioni (C5) perdono 0.006 rispetto a C1 sulle clean (0.818 contro 0.824) e 0.007 su tutte le righe: perdita piccola, coerente con Q2, ma gli IC si sovrappongono molto.
- Calibrazione: l'ECE grezza di C1/C5 è 0.056/0.060, peggiore della CV (0.031 per SHD, Q5), coerente con la sovrastima in popolazioni a minor prevalenza. Per probabilità assolute serve ricalibrare (non fatto sul test, per non ottimizzare); PPV/NPV dipendono dalla prevalenza.
- Limiti: un solo centro, solo pazienti con eco, nessuna validazione esterna. Nessun sottogruppo valutato sul test.

## Decisioni e motivazioni
- Confermati 5 su 5 secondo i criteri fissati; nessuna soglia modificata, nessun candidato aggiunto o tolto, nessuna ricalibrazione sul test.
- Popolazione del criterio = clean (regola 4), scelta fatta prima dell'apertura del test ma non scritta nella preregistrazione.
- Classe «buona» per tutti (AUROC clean ≥ 0.80, IC inf ≥ 0.75). La classe è definitiva solo sul test e questa è una seconda lettura.
- Alternativa scartata: confrontare il tabulare (tutte le righe) con i valori clean: popolazioni diverse.

## Configurazioni provate
0 nuove configurazioni (nessun log-exp in questa fase), 0 con uso del validation. Candidati valutati sul test: 5. Registro prima di Q8: 725 configurazioni (pipeline 1 e 2), di cui 214 nella pipeline 2. Il test è stato letto una volta in P7 e una volta in Q8.

## Artefatti
- `results/final2/test_results.json`
- `reports/final_test2_table.md`
- `reports/Q8_final-test2.progress.md`

## Problemi aperti
- Primo tentativo di `final_eval2.py` con il Python di sistema fallito (`ModuleNotFoundError: torch`) dopo il caricamento del test in memoria, senza metriche né file; rilanciato con `.venv`. Dichiarato per trasparenza.
- Seconda lettura del test, con pipeline progettata da chi conosceva P7: le conferme sono meno indipendenti di una prima lettura.
- Popolazione del criterio non preregistrata (fissata con la regola 4); riferimento CV di C5 a seed singolo.
- Il tabulare è su tutte le righe, non clean; assente per C3 e C4.
- ECE e Brier solo su tutte le righe e con probabilità grezze.
- AUROC del test sopra la CV: non interpretare come miglioramento reale.
- Un solo centro, solo pazienti con eco, nessuna validazione esterna; SHD non è scompenso cardiaco.

## Handoff
STATUS: OK
RISULTATI_CHIAVE:
- TABELLA_TEST (clean, IC95% per paziente): C1 SHD 0.824 [0.813-0.836]; C2 LVEF≤45 0.897 [0.886-0.908]; C3 VD 0.889 [0.873-0.904]; C4 stenosi 0.832 [0.811-0.857]; C5 SHD 3 deriv. 0.818 [0.808-0.829]
- TUTTE_LE_RIGHE: C1 0.837, C2 0.905, C3 0.899, C4 0.841, C5 0.830; vs tabulare: C1 +0.075 [+0.063,+0.086], C2 +0.112 [+0.099,+0.127], C5 +0.068 [+0.057,+0.081]; C3/C4 senza comparatore
- CONFERMATI: [C1_shd_all, C2_lvef45_all, C3_rv_all, C4_as_all, C5_shd_I_II_V1]
- NON_CONFERMATI: []
- CALO_CV_TEST: C1 -0.009, C2 -0.028, C3 -0.045, C4 -0.021, C5 -0.016 (negativo = test sopra la CV, non evidenza di miglioramento)
- CLASSE: C1 buona, C2 buona, C3 buona, C4 buona, C5 buona
- ECE/BRIER (tutte le righe, grezze): C1 0.056/0.163, C2 0.025/0.081, C3 0.017/0.051, C4 0.008/0.044, C5 0.060/0.167
- Seconda lettura del test (prima P7: CNN SHD 0.830, tab. LVEF≤45 0.793); 5 candidati; primo tentativo fallito per torch mancante senza metriche
RACCOMANDAZIONE:
- Q9: riportare i 5 come confermati con le cautele sopra (seconda lettura, popolazione non preregistrata, test sopra CV da non enfatizzare, ECE peggiore della CV, SHD diverso da scompenso).
- Non usare di nuovo il test: due letture già consumate.
