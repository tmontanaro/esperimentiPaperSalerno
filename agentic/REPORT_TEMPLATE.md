# Template del report di fase

Ogni fase produce `reports/<ID>_<nome>.md` con ESATTAMENTE queste sezioni `##` (il validatore le controlla).
Scrivi in italiano, per un lettore che non è esperto di ML. Un report è buono se un'altra persona (o un altro agente con il contesto vuoto)
può ripartire da lì senza chiedere nulla.

```markdown
# Report <ID> · <nome>
- Agente: <nome agente>
- Data: <AAAA-MM-GG HH:MM>
- Stato: OK | PARZIALE | FALLITO

## Obiettivo
Cosa doveva essere stabilito in questa fase e perché è utile (1-3 frasi).

## Input letti
Elenco dei file/Handoff usati (percorsi), con eventuali valori ereditati dalle fasi precedenti.

## Cosa è stato fatto
Passi in ordine, con i percorsi degli script in experiments/<ID>/. Cosa è stato fatto in smoke test (debug_n) e cosa sull'intero dataset.

## Risultati
Tabelle con numeri: metrica, IC95%, n, n positivi, confronto con il riferimento. Ogni affermazione ha un numero.
Indica sempre su quale split è calcolato ogni valore (train/CV, validation).

## Decisioni e motivazioni
Scelte fatte (modello, soglie, feature scartate) e perché. Alternative scartate.

## Configurazioni provate
Quante configurazioni/regole/modelli sono stati provati (numero) e quante hanno usato il validation per scegliere.
Devono coincidere con le righe registrate con `tools/state.py log-exp`.

## Artefatti
Percorsi dei file prodotti (CSV, figure, candidati in results/<ID>/candidates/...).

## Problemi aperti
Anomalie, limiti, cose non verificate, dipendenze mancanti. Scrivi "Nessuno" solo se è vero.

## Handoff
STATUS: OK|PARZIALE|FALLITO
RISULTATI_CHIAVE:
- (massimo 8 righe, ciascuna con un numero; usa le chiavi richieste dalla tua fase nel PLAYBOOK)
RACCOMANDAZIONE:
- (cosa dovrebbe fare/decidere la fase successiva, in 1-4 righe)
```

Regole sull'Handoff: massimo 40 righe non vuote; è la parte che le altre fasi leggono al posto del report intero.
Se non sei riuscito a completare la fase usa STATUS: PARZIALE e spiega in "Problemi aperti" cosa manca e da dove ripartire.
Usa STATUS: FALLITO solo se la fase non può proseguire (es. dati mancanti o corrotti).
