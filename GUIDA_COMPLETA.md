# Guida completa: pipeline multi-agente EchoNext (Claude Code) — Parte 1 (pipeline 1) e Parte 2 (pipeline 2)

Questa guida porta il workflow EchoNext (i 5 notebook) in una pipeline in cui **agenti specializzati lavorano uno dopo l'altro**, un **orchestratore** decide cosa fare in base ai risultati,
un **revisore indipendente** cerca gli errori e alla fine esce un **report spiegato**. Ogni passaggio viene scritto su disco, quindi se il processo si ferma (token finiti, crash, chiusura del computer) **riparti da dove eri, non dall'inizio**.

> **Come usare questa guida.** La **Parte 1** (sezioni 0-9) è la guida della pipeline 1 (fasi P0-P9), invariata. La **Parte 2** (sezioni 10-19, in fondo) guida la **pipeline 2** (fasi Q0-Q10), che si aggancia ai risultati della pipeline 1 e fa i test ancora possibili con gli stessi dati.
> Chi parte da zero esegue prima la Parte 1 e poi la Parte 2. Chi ha già completato la pipeline 1 salta direttamente alla sezione 10.

## Indice
0. Prima di iniziare: tre verifiche (leggile davvero)
1. Installare Claude Code
2. Preparare il progetto
3. Come funziona (in due minuti)
4. Prova a secco: collaudare la pipeline prima dei dati veri
5. Eseguire sui dati reali
6. Se si blocca: ripartire
7. Leggere i risultati e approvare il test
8. Personalizzare
9. Cosa non è stato verificato

**Parte 2 — pipeline 2**
10. Che cosa fa la pipeline 2 e perché
11. Prima di iniziare: prerequisiti e avvertenze (leggile davvero)
12. Installare i file nuovi
13. Prova a secco della pipeline 2
14. Eseguire la pipeline 2 sui dati reali
15. La pausa prima del secondo test (approvazione)
16. Se si blocca: ripartire
17. Leggere i risultati, il report complessivo e il notebook
18. Cosa non è stato verificato (pipeline 2)
19. Mappa dei file nuovi

---

## 0. Prima di iniziare: tre verifiche

**A. Serve un account che includa Claude Code.** Secondo la documentazione ufficiale serve un account Pro, Max, Team, Enterprise o Console; **il piano gratuito di claude.ai non include Claude Code**.
In alternativa si può usare un provider API di terze parti (Amazon Bedrock, Google Cloud, Microsoft Foundry). Chiedi al tuo ateneo se ha un accordo.

**B. Verifica che i dati EchoNext possano passare da un servizio cloud. Questo è il punto più importante.**
Quando un agente esegue codice sui tuoi dati, ciò che il codice stampa (e ciò che i file leggibili contengono) viene inviato ad Anthropic per essere elaborato.
- PhysioNet, nel suo avviso del 24 settembre 2025 «Use of MIMIC Data with Large Language Models and Online Services», scrive che il Data Use Agreement *vieta di condividere l'accesso ai dati con terze parti, inclusi l'invio tramite API o l'uso su piattaforme online*.
  Per i servizi cloud chiede di verificare zero data retention, nessun uso dei dati per il training e nessuna revisione umana, avverte che i requisiti dichiarati dai servizi possono non bastare e **raccomanda fortemente modelli LLM in locale**.
- Quell'avviso cita MIMIC e derivati, **non EchoNext**. EchoNext ha il suo accordo d'uso (la licenza è nella pagina del dataset): **leggilo** e chiedi conferma scritta al responsabile del progetto o dell'ufficio dati.
- Non ho verificato le condizioni di conservazione dei dati di Anthropic per il tuo tipo di account: controllale tu.
- Il kit **riduce** l'esposizione (gli agenti sono istruiti a stampare solo aggregati, non possono aprire i file dati con lo strumento di lettura, `lib.show()` rifiuta tabelle a livello di paziente), ma non la azzera: codice scritto da un agente potrebbe stampare altro.
- Se non hai la conferma, **fermati al passo 4**: la prova completa si fa su dati sintetici e non ha alcun rischio.

**C. Risorse.** Le forme d'onda pesano circa 12 GB (spazio su disco libero doppio se estrai da uno zip). Per la fase P4 (CNN) serve una GPU NVIDIA: senza, la fase fa solo una prova e dice come lanciarla altrove.
Le fasi tabulari girano su un portatile.

---

## 1. Installare Claude Code

Comandi ufficiali (verificati sulla documentazione, novembre 2026):

| Sistema | Comando |
|---|---|
| Linux, macOS, WSL | `curl -fsSL https://claude.ai/install.sh \| bash` |
| Windows PowerShell | `irm https://claude.ai/install.ps1 \| iex` |
| Homebrew | `brew install --cask claude-code` |
| npm (serve Node.js 22+) | `npm install -g @anthropic-ai/claude-code` (mai con `sudo`) |

1. Apri un **nuovo** terminale e verifica: `claude --version` (stampa un numero di versione). Diagnostica: `claude doctor`.
2. Primo avvio e accesso: nella cartella del progetto digita `claude` e segui le istruzioni nel browser. Se compare «command not found», la cartella di installazione non è nel PATH: vedi la pagina *Troubleshoot installation* della documentazione.
3. Requisiti: Ubuntu 20.04+, Debian 10+, macOS 13+, Windows 10 (1809+), almeno 4 GB di RAM. Su Linux gli aggiornamenti sono automatici (installazione nativa).
4. Documentazione: https://code.claude.com/docs

---

## 2. Preparare il progetto

```bash
unzip echonext-agentic.zip          # oppure: python3 -m zipfile -e echonext-agentic.zip .
cd echonext-agentic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch: con GPU NVIDIA usa la build CUDA indicata su pytorch.org
git init && git add -A && git commit -m "kit iniziale"    # consigliato: permette di tornare indietro e il driver salva un commit dopo ogni fase
```

I dati **restano fuori** dalla cartella del progetto (es. `/home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1/`): il kit li trova anche in sottocartelle o dentro uno zip.
Se il caricamento non trova il CSV, l'errore elenca cosa c'è nella cartella.

---

## 3. Come funziona (in due minuti)

```
   ┌────────────┐  gate   ┌────────────┐  gate   ┌────────────┐         ┌──────────┐  ok umano  ┌────────┐   ┌────────────┐
   │ P0 audit   │ ──────▶ │ P1 baseline│ ──────▶ │ P2 / P3 /  │ ──────▶ │ P6 review│ ─────────▶ │P7 test │──▶│ P8 sintesi │
   │ dati       │         │ tabulare   │         │ P4 / P5    │         │ scettico │            │ (1 volta)│  │ report     │
   └────────────┘         └────────────┘         └────────────┘         └──────────┘            └────────┘   └────────────┘
        ▲  ogni fase = un agente, una sessione nuova, un report su disco  ▲  il gate (orchestratore) legge l'Handoff e decide: salta, aggiunge, riordina
```

- **Memoria = file.** Nessun agente si "ricorda" nulla: ognuno comincia con `python3 tools/state.py brief`, che stampa stato, decisioni e gli *Handoff* delle fasi precedenti.
- **Un report per fase** (`reports/Pn_nome.md`), sempre con le stesse sezioni (Obiettivo, Input letti, Cosa è stato fatto, Risultati, Decisioni, Configurazioni provate, Artefatti, Problemi aperti, **Handoff**). Un validatore rifiuta i report incompleti.
- **Le decisioni adattive** sono nel `PLAYBOOK` (sezioni «Gate»): se le sole misure tabulari non portano segnale, le forme d'onda diventano prioritarie e il clustering si riduce; se il QRS risulta un modificatore d'effetto, parte un'analisi stratificata; ecc.
- **Protezioni imposte dal codice**: il test set non si apre fuori da P7 e solo una volta; le colonne derivate dall'eco sono rifiutate come input; i candidati si valutano con un contratto fisso.

Mappa dei file:

| Percorso | Contenuto |
|---|---|
| `CLAUDE.md` | regole di progetto lette da tutti gli agenti |
| `agentic/PLAYBOOK.md` | fasi, soglie, definizioni di «credibile», regole dei gate (**qui decidi tu**) |
| `agentic/pipeline_state.json` | stato (si modifica solo con `tools/state.py`) |
| `agentic/DECISIONS.md`, `agentic/RUNLOG.md` | registro delle decisioni e log di ogni chiamata (esito, durata, costo stimato) |
| `agentic/backups/` | copia dello stato prima e dopo ogni fase |
| `reports/` | un report per fase + `*.progress.md` (appunti per riprendere a metà fase) |
| `results/`, `results/registry.csv` | output, candidati, elenco di ogni configurazione provata |
| `.claude/agents/`, `.claude/commands/` | gli agenti e i comandi `/run-pipeline`, `/resume`, `/status` |
| `legacy_notebooks/` | i notebook di partenza (ricette per gli agenti) |
| `BUILD_PROGRESS.md` | note su come è stato costruito il kit |

---

## 4. Prova a secco: collaudare la pipeline prima dei dati veri

**4A. Simulazione senza token (2 minuti).** Un finto `claude` simula gli agenti e verifica che stato, ripresa, pause e controlli funzionino sul tuo computer.

```bash
python3 tools/make_synth.py /tmp/echonext_synth 3000        # dati sintetici con lo stesso schema (nessun valore scientifico)
python3 tools/state.py init --data-dir /tmp/echonext_synth
python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py ; echo "codice di uscita: $?"
```
Il driver si ferma con codice 10 prima di P7 (serve la tua approvazione). Procedi:
```bash
python3 tools/state.py approve P7
python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py
```
Per provare i guasti: `FAKE_FAIL_ONCE=P2 FAKE_LIMIT_ON=P3 FAKE_BAD_REPORT=P4 python3 tools/run_pipeline.py --claude-bin tools/fake_claude.py`
(errore, limite d'uso e report incompleto: il driver riprova, esce con 75 al limite e alla riesecuzione riparte da P3).
Poi: `python3 tools/reset.py --yes`.

**4B. Prova con Claude vero sui dati sintetici (costa pochi token).** Verifica che i permessi e i prompt funzionino davvero e non c'è alcun rischio sui dati.
```bash
python3 tools/state.py init --data-dir /tmp/echonext_synth --force
python3 tools/run_pipeline.py --until P1 --max-budget-usd 3
```
Leggi `reports/P0_data-audit.md`, `reports/P1_baseline.md`, `agentic/RUNLOG.md`. Se un agente è stato bloccato dai permessi (in modalità headless ciò che non è consentito viene negato senza chiedere), lo vedrai nei report in «Problemi aperti»: aggiusta `.claude/settings.json`.
Poi `python3 tools/reset.py --yes`.

---

## 5. Eseguire sui dati reali (solo dopo la verifica B del passo 0)

```bash
python3 tools/reset.py --yes
python3 tools/state.py init --data-dir /home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1
```
`init` scrive anche una regola in `.claude/settings.local.json` che impedisce agli agenti di aprire i file della cartella dati con lo strumento di lettura.

**Controllo obbligatorio prima di proseguire:** `python3 tools/state.py brief | head -5`. Deve mostrare il TUO `data_dir` (non `/tmp/echonext_synth`) e tutte le fasi `pending`. Se vedi fasi già `done`, o un Handoff con «risultato simulato», lo stato è un residuo delle prove: ripeti `reset` e `init --force` (e `rm -f .claude/settings.local.json`). Segnale d'allarme sui risultati: numeri identici a quelli della prova sintetica (es. AUROC 0.574 con età+sesso 0.588) = stai ancora lavorando sui dati finti.

**Passo obbligatorio prima di tutto: approva la cartella.** Apri `claude` una volta nella cartella del progetto (`cd echonext-agentic && claude`) e accetta la richiesta «Do you trust this folder?». Senza questo passo, in modalità non interattiva (`-p`, usata dal driver) Claude Code **ignora le regole di permesso** del progetto e gli agenti non possono lavorare: il driver se ne accorge e termina con codice 3. Poi esci (`/exit`).

### Modo A: interattivo (consigliato la prima volta)
```bash
claude
> /run-pipeline            # oppure: /run-pipeline --until P3
```
La sessione principale fa da orchestratore e lancia un sottoagente per fase; vedi tutto e puoi intervenire. Se modifichi un file in `.claude/agents/` riavvia `claude`, perché gli agenti si caricano all'avvio.

### Modo B: non interattivo (lascialo lavorare)
```bash
python3 tools/run_pipeline.py                    # una sessione nuova per fase, tutto su disco
nohup python3 tools/run_pipeline.py > run.log 2>&1 &
tail -f run.log                                  # Ctrl+C esce dal tail, non ferma la pipeline
```
Per riprovare da solo quando finisce il limite d'uso (codice di uscita 75), ogni 30 minuti:
```bash
until python3 tools/run_pipeline.py; do
  rc=$?
  [ "$rc" -eq 75 ] || break
  echo "limite raggiunto: riprovo tra 30 minuti"; sleep 1800
done
```
Codici di uscita: **0** finita · **2** fase fallita o bloccata · **4** programma `claude` non trovato (vedi sotto) · **3** cartella non approvata (vedi sopra) · **10** attende la tua approvazione · **75** limite d'uso.

Se vedi «Non trovo il programma `claude`» (o `FileNotFoundError: ... 'claude'`): il driver non lo trova nel PATH del terminale in cui lo lanci (tipico con un ambiente virtuale `.venv` attivo o con installazione in `~/.local/bin`). Controlla con `which claude`; se non risponde, installa Claude Code (passo 1) o aggiungi `~/.local/bin` al PATH (`export PATH="$HOME/.local/bin:$PATH"`), oppure usa `--claude-bin /percorso/di/claude`.
Opzioni utili: `--until P3`, `--model sonnet`, `--max-turns`, `--max-budget-usd`, `--timeout-min`, `--no-gates`, `--dry-run`.

---

## 6. Se si blocca: ripartire

**Regola d'oro: non ricominciare mai da capo.** Lo stato è in `agentic/pipeline_state.json` e i report sono in `reports/`.

| Cosa è successo | Cosa fare |
|---|---|
| Limite di token/uso a metà fase | Il driver esce con 75 e rimette la fase in coda senza contare il tentativo. Rilancia lo stesso comando. L'agente riparte da `reports/<ID>_<nome>.progress.md` e da `results/<ID>/`. |
| Sessione interattiva piena o chiusa | Apri `claude` e digita `/resume`. Fa `brief`, recupera le fasi interrotte, chiude i gate dovuti e continua. |
| Il computer si è spento o il processo è stato ucciso | `python3 tools/run_pipeline.py` all'avvio rimette in coda le fasi rimaste «running». In interattivo: `/resume`. |
| Una fase è fallita 3 volte | Il driver si ferma con codice 2. Leggi `agentic/RUNLOG.md` e il report; correggi la causa; poi `python3 tools/state.py reset <ID> --attempts` e rilancia. |
| Vuoi rifare una fase | `python3 tools/state.py reset <ID> --attempts`, poi rilancia. Le fasi successive già concluse restano: azzerale con lo stesso comando se dipendono dalla fase rifatta. |
| Lo stato sembra rovinato | Ripristina da `agentic/backups/<data>_prima_<fase>/pipeline_state.json` o con `git log` / `git checkout <commit> -- agentic reports results`. |

Se i comandi `/resume` non sono disponibili, incolla questo prompt in una nuova sessione di Claude Code dentro la cartella del progetto:
```
Riprendi la pipeline EchoNext. Esegui `python3 tools/state.py brief`. Se ci sono fasi 'running' esegui `python3 tools/state.py recover`.
Applica i gate dovuti descritti in agentic/PLAYBOOK.md solo con tools/state.py. Poi continua la fase successiva con il sottoagente indicato nel campo `agent`.
Non rifare ciò che è già in reports/ e results/. Non aprire il test set. Fermati prima di P7 e chiedimi l'approvazione.
```

---

## 7. Leggere i risultati e approvare il test

Ordine di lettura consigliato:
1. `reports/P8_synthesis.md` (copia in `FINAL_REPORT.md`): la risposta spiegata, con numeri e limiti.
2. `reports/P6_review.md`: cosa ha trovato il revisore (leakage, numeri non riproducibili, affermazioni deboli).
3. `agentic/DECISIONS.md`: perché l'orchestratore ha aggiunto o saltato fasi. `results/registry.csv`: quante configurazioni sono state provate (più sono, più cautela).
4. I report delle singole fasi, partendo dagli *Handoff*.

**Prima del test (P7)** il driver si ferma. Guarda `agentic/FROZEN.json` (quali candidati) e `reports/P6_review.md` (verdetto). Se ti convincono: `python3 tools/state.py approve P7` e rilancia.
Il test si apre **una sola volta**: se un candidato va male lo si riporta così com'è, perché ritoccare il modello dopo aver visto il test significherebbe selezionare sul test.

**Per riportare ai partner**, usa le classi del PLAYBOOK: «buona» (AUROC test ≥ 0.80 con limite inferiore dell'IC ≥ 0.75), «moderata» (0.70-0.80), «debole». Se non ci sono candidati credibili,
il risultato è esso stesso utile: le sole misure ECG tabulari non bastano e la strada sono le forme d'onda. Ricorda sempre che SHD non coincide con scompenso cardiaco.

---

### Fase finale P9: notebook per i partner
L'ultima fase (agente `notebook-builder`) genera `reports/P9_notebook/replica_analisi.ipynb`: un notebook autonomo che rifà passo passo le analisi, con obiettivo, risultato atteso e decisione presa in ogni capitolo (e la regola di decisione ricalcolata, così si vede come un risultato guida il passo successivo). Il test set non è incluso; cartella dati in un parametro (`DATA_DIR`), modalità `QUICK` per provarlo in pochi minuti. Accanto ci sono `LEGGIMI.md` e `requirements.txt`.
- Con una pipeline appena inizializzata la fase c'è già.
- Se hai già inizializzato prima di questo aggiornamento, aggiungila una volta sola (dopo aver copiato `.claude/agents/notebook-builder.md`):
```bash
python3 tools/state.py set-config max_followups 5
python3 tools/state.py add-phase --id P9 --name notebook --agent notebook-builder --depends-on P8 --goal "Notebook Jupyter per i partner"
```
- Per eseguire solo questa fase a pipeline finita: `python3 tools/run_pipeline.py` (parte da P9).

## 8. Personalizzare

- **Soglie**: `python3 tools/state.py set-config thresholds.good_auroc 0.85` (vedi l'elenco con `python3 tools/state.py status --json`). Le *definizioni* si cambiano in `agentic/PLAYBOOK.md`.
- **Modelli e costi**: in ogni `.claude/agents/*.md` la riga `model:` accetta `sonnet`, `opus`, `haiku`. Il revisore usa `opus`, gli altri `sonnet`. Un modello più piccolo costa meno ma sbaglia di più; non abbassare il revisore. Per forzarlo ovunque: `--model sonnet`.
- **Limiti**: `maxTurns` nei file degli agenti, `--max-turns` e `--max-budget-usd` nel driver.
- **Aggiungere un agente**: crea `.claude/agents/nome.md` (frontmatter `name`, `description`, `tools`, `model`), aggiungi la sua sezione al PLAYBOOK e inseriscilo con
  `python3 tools/state.py add-phase --id P9 --name nome --agent nome --goal "..." --depends-on P6 --gate`.
- **Permessi**: `.claude/settings.json` (cosa gli agenti possono scrivere e lanciare). In headless ciò che non è nell'elenco viene negato.
- **Obiettivo diverso** (es. un solo target): cambia `config.primary_target` e le sezioni del PLAYBOOK.

---

## 9. Cosa non è stato verificato

- **Collaudo con Claude reale (parziale).** Su dati sintetici sono state eseguite con la CLI vera le fasi **P0 (audit)** e **P1 (baseline)** con i relativi gate (circa 0,6 USD in tutto): gli agenti hanno prodotto report validi, usato `lib.show()`, registrato le configurazioni con `log-exp`, salvato il candidato e segnalato correttamente «nessun segnale» (i dati sintetici non ne hanno). **Le fasi P2–P8 sono state provate solo con un finto `claude`** (stato, driver, ripresa dopo errore/limite/report incompleto, pausa per approvazione, blocco del test set, valutazione finale, snapshot, checkpoint git). I loro prompt vanno quindi verificati con il passo 4B e probabilmente rifiniti dopo la prima esecuzione vera. Costi e tempi su dati reali (e CNN su GPU in P4) non sono stimati.
- L'output JSON di `claude -p` (`result`, `session_id`, `total_cost_usd`, `is_error`) è letto in modo tollerante: i costi mostrati in `RUNLOG.md` sono una stima e possono risultare `n/d`.
- Le soglie del PLAYBOOK sono valori iniziali ragionevoli, non standard: discutile con i colleghi.
- Gli agenti sono LLM: possono sbagliare o inventare. Per questo ci sono il revisore, le protezioni nel codice e il registro delle configurazioni, ma **la lettura finale spetta a una persona** (e, per il significato clinico, a un cardiologo).
- Informazioni su Claude Code (installazione, subagenti, comandi, modalità headless) e sulla politica PhysioNet: documentazione consultata il 6 novembre 2026; possono cambiare.

---

# PARTE 2 · Pipeline 2 (fasi Q0-Q10)

## 10. Che cosa fa la pipeline 2 e perché

La pipeline 1 ha trovato che le sole 7 misure ECG riassuntive danno una predizione moderata, che una rete neurale (CNN) sulle forme d'onda ne dà una buona, che nessun sottogruppo o cluster basato sulle misure tabulari è credibile e che l'ipotesi dei medici sul QRS è inconclusiva.
Restavano però domande a cui si può rispondere **con gli stessi dati** (nessun dato nuovo né informazione mancante è disponibile):

| Fase | Domanda | In parole povere |
|---|---|---|
| Q0 | I presupposti reggono? | Controlla cosa esiste davvero, verifica l'ordine delle derivazioni ECG (PhysioNet non lo documenta), misura i tempi di calcolo. |
| Q1 | La CNN è solida? | Rifà la CNN più volte (3 addestramenti diversi) e in cross-validation per paziente su train+validation, così si vede quanto il risultato dipende dalla fortuna. |
| Q2 | Quali derivazioni servono? | Toglie derivazioni a gruppi (l'idea originale di togliere input) per vedere se bastano meno tracciati. |
| Q3 | Il QRS conta per la LVEF? | Rifà l'ipotesi dei medici usando la CNN, che prevede molto meglio delle 7 misure. |
| Q4 | Esistono gruppi di pazienti diversi? | Cluster e sottogruppi sulla rappresentazione interna della CNN, con confronto contro il «rumore di fondo». |
| Q5 | Le probabilità sono affidabili? Cosa guarda la rete? | Calibrazione, soglie operative, derivazioni e istanti più importanti. |
| Q6 | Quali modelli portiamo al secondo test? | Regole scritte *prima*, addestramento e congelamento dei candidati (con impronte che impediscono di cambiarli dopo). |
| Q7 | Si regge a una critica indipendente? | Un revisore scettico riproduce i numeri con codice suo e può solo togliere candidati. |
| Q8 | Conferma sul test | Seconda (e ultima) lettura del test, pre-specificata, su pochi candidati. **Serve la tua approvazione.** |
| Q9 | Cosa è emerso? | **Report complessivo in parole povere**: obiettivo, prove fatte passo passo, risultati, cosa è fattibile e cosa no. |
| Q10 | Come lo si rifà? | **Notebook complessivo** che integra quello della pipeline 1 e aggiunge i capitoli nuovi. |

Principi: le scelte si fanno solo con la cross-validation (mai guardando il test); il test si legge una volta sola anche qui; ogni numero ha un intervallo di confidenza calcolato per paziente; si riporta anche ciò che non funziona.

## 11. Prima di iniziare: prerequisiti e avvertenze (leggile davvero)

1. **La pipeline 1 deve essere completa sui dati reali** (fasi P0-P9 `done`): `python3 tools/state.py brief`. La pipeline 2 prende da lì la cartella dei dati e le soglie.
2. **NON lanciare `python3 tools/reset.py` (quello della Parte 1)**: cancella report, risultati e stato della pipeline 1. Per azzerare la pipeline 2 esiste un comando separato: `python3 tools/reset2.py --yes`, che tocca solo i file con prefisso Q.
3. **Fai una copia di sicurezza** prima di iniziare:
   ```bash
   tar czf ../backup_pipeline1_$(date +%Y%m%d_%H%M).tgz agentic reports results experiments .claude
   ```
4. **Hardware**: serve una GPU (la CNN viene addestrata decine di volte). Con la GPU RTX 5080 usata finora un addestramento completo dura circa un minuto; Q0 misura i tempi sul tuo computer e stima la durata di Q1, Q2 e Q6. Su CPU la pipeline 2 non è praticabile.
5. **Disco**: ogni addestramento salva i pesi (pochi MB) e le predizioni; in tutto qualche GB al massimo. Controlla con `df -h .`.
6. **Account e dati**: valgono le avvertenze della sezione 0 (account a pagamento, regole PhysioNet/DUA di EchoNext). Nei report e nell'output passano solo numeri aggregati.
7. **Cartella già approvata** in Claude Code (sezione 5): non serve rifarlo.
8. Il test set resta chiuso. Verrà letto una sola volta in Q8, solo dopo la tua approvazione.

## 12. Installare i file nuovi

Copia i file del pacchetto nella cartella del progetto **mantenendo i percorsi** (lo zip `echonext-agentic.zip` li contiene già; se ti sono arrivati singolarmente vanno messi così):

```
lib/pipe2_lib.py
tools/cnn_cv.py  tools/state2.py  tools/phases2.py  tools/run_pipeline2.py  tools/final_eval2.py  tools/reset2.py
tools/run_pipeline.py                 <- SOSTITUISCE il precedente (aggiunge solo tre messaggi parametrizzabili)
agentic/PLAYBOOK2.md
.claude/agents/   inventory-auditor.md  cnn-cv-analyst.md  lead-ablation-analyst.md  lvef-qrs-cnn-analyst.md  embedding-analyst.md
                  calibration-analyst.md  candidate-freezer.md  skeptic-reviewer2.md  final-evaluator2.md  synthesizer2.md  notebook-builder2.md
.claude/commands/run-pipeline2.md
GUIDA_COMPLETA.md
```
Controllo: `python3 tools/state2.py --help` deve mostrare l'elenco dei comandi senza errori, e `python3 -c "import sys; sys.path.insert(0,'lib'); import pipe2_lib, torch; print('ok')"` deve stampare `ok`.
Se avevi già una sessione di `claude` aperta, riavviala: gli agenti si caricano all'avvio.

## 13. Prova a secco della pipeline 2

Serve a verificare che stato, ripresa e pausa funzionino sul tuo computer, **senza spendere token e senza toccare i tuoi risultati**: si fa in una **copia** della cartella.
```bash
cd ..
cp -r echonext-agentic echonext-agentic_prova && cd echonext-agentic_prova
python3 tools/state2.py init
python3 tools/run_pipeline2.py --claude-bin tools/fake_claude.py ; echo "codice di uscita: $?"      # si ferma con 10 prima di Q8
python3 tools/state2.py approve Q8
python3 tools/run_pipeline2.py --claude-bin tools/fake_claude.py ; echo "codice di uscita: $?"      # arriva a «Pipeline 2 completata»
cd .. && rm -rf echonext-agentic_prova
```
Il finto `claude` scrive report segnati «SIMULATO»: non valgono come risultati. (Se la copia occupa troppo spazio, escludi `results` con `rsync -a --exclude results`.)

## 14. Eseguire la pipeline 2 sui dati reali

Dalla cartella vera del progetto, con l'ambiente virtuale attivo:
```bash
python3 tools/state2.py init            # prende dati e soglie dalla pipeline 1 e protegge i file della pipeline 2
python3 tools/state2.py brief | head -15   # controlla: data_dir giusto, fasi Q0-Q10 tutte pending
python3 tools/run_pipeline2.py --max-budget-usd 5 --timeout-min 240
```
Lascialo lavorare. Il driver stampa **una riga per fase** quando finisce: durante Q1 e Q2 (le più lunghe) il terminale resta fermo per decine di minuti, ed è normale.
Per riprovare da solo dopo un limite d'uso (codice 75):
```bash
until python3 tools/run_pipeline2.py --max-budget-usd 5 --timeout-min 240; do
  rc=$?
  [ "$rc" -eq 75 ] || break
  echo "limite raggiunto: riprovo tra 30 minuti"; sleep 1800
done
```
Codici di uscita (come nella Parte 1): **0** finita · **2** fase fallita o bloccata · **3** cartella non approvata · **4** `claude` non trovato · **10** attende la tua approvazione (Q8) · **75** limite d'uso.

**Come sapere che sta lavorando** (da un secondo terminale, nella cartella del progetto):
```bash
python3 tools/state2.py brief | sed -n 5,18p           # fase in corso (running) e completate
cat results/Q1/main/progress.json | head -20           # job completati/totali, epoca, ora dell'ultimo aggiornamento (anche per Q2: results/Q2/<insieme>/)
ls -lt results/Q* reports | head                       # file modificati di recente
nvidia-smi                                             # la GPU è occupata durante gli addestramenti
```
Se `updated` in `progress.json` non cambia da più di 15 minuti, la GPU è ferma e `brief` mostra la stessa fase da molto tempo: interrompi con Ctrl+C e rilancia lo stesso comando (riparte dal job successivo).
Gli agenti lanciano l'addestramento «a pezzi» da circa 7 minuti e lo rilanciano finché stampa `STATO: COMPLETO`; vedrai quindi anche `STATO: PARZIALE` nei log: è normale, non è un errore.

## 15. La pausa prima del secondo test (approvazione)

Dopo Q7 il driver si ferma con codice 10. **Prima di approvare guarda:**
```bash
python3 tools/state2.py handoff Q6          # candidati scelti e criteri di conferma
python3 tools/state2.py handoff Q7          # verdetto del revisore: PASS/REVISE, candidati ammessi
cat agentic/FROZEN2.json                    # candidati che andranno al test (con le impronte)
cat results/Q6/preregistration.md           # regole scritte PRIMA di guardare il test
```
Verifica che i candidati siano pochi (al massimo 6) e che il revisore non abbia segnalato problemi gravi. Poi:
```bash
python3 tools/state2.py approve Q8
python3 tools/run_pipeline2.py --max-budget-usd 5 --timeout-min 240
```
Garanzie nel codice: il test si legge **una sola volta** (una seconda esecuzione viene rifiutata); se un candidato o i suoi pesi sono stati modificati dopo il congelamento, `final_eval2.py` si ferma; il test era già stato letto in P7, quindi questa è dichiarata come *seconda lettura* e il report lo scrive.

## 16. Se si blocca: ripartire

Vale tutto ciò che è scritto nella sezione 6. In più, per la pipeline 2:
- Usa sempre `tools/state2.py` e `tools/run_pipeline2.py`, mai quelli della Parte 1.
- Dopo un'interruzione: `python3 tools/state2.py recover` e rilancia `python3 tools/run_pipeline2.py ...`. La fase interrotta riparte; **i job della CNN già fatti non si rifanno** (ogni coppia seed×fold ha il suo file).
- Se un addestramento è stato interrotto a metà, riparte quel solo job. Se cambi parametri (epoche, derivazioni...) in una cartella già usata, il programma lo rifiuta per non mischiare run diverse: usa un'altra cartella `--out`.
- Fase fallita: `python3 tools/state2.py reset Qn --attempts` e rilancia.
- Azzerare solo la pipeline 2: `python3 tools/reset2.py --yes` (non tocca la pipeline 1; non lanciare mai `tools/reset.py`).
- I backup dello stato sono in `agentic/backups2/` (uno prima e uno dopo ogni fase).
- Modalità interattiva (opzionale): in `claude`, `/run-pipeline2`.

## 17. Leggere i risultati, il report complessivo e il notebook

1. **`reports/REPORT_COMPLESSIVO.md`**: il documento da leggere per primo e da usare con i colleghi. Spiega in parole povere (a) l'obiettivo, (b) i risultati, (c) le prove fatte passo passo (prima pipeline 1, poi pipeline 2), (d) cosa è fattibile e cosa no, (e) cosa si può ancora fare con questi dati, (f) quanto fidarsi. Ne esiste anche la versione con il formato dei report: `reports/Q9_synthesis2.md`.
2. **`reports/final_test2_table.md`** e `reports/Q8_final-test2.md`: la seconda lettura del test.
3. **Un report per fase**: `reports/Q0_inventory.md` … `reports/Q7_review2.md`. Le sezioni «Problemi aperti» dicono cosa non è andato bene.
4. **`reports/Q10_notebook/replica_completa.ipynb`** (con `LEGGIMI.md` e `requirements.txt`): notebook che include quello della pipeline 1 e aggiunge i capitoli Q0-Q8. Si apre con Jupyter (`pip install jupyterlab`, poi `jupyter lab`); compila `DATA_DIR` nella prima cella, lascia `QUICK=True` per una prova rapida e `RUN_HEAVY=False` per saltare gli addestramenti. Per rifare davvero le parti pesanti serve la GPU e `RUN_HEAVY=True`.
   Il notebook non contiene il test: riporta come testo la tabella del secondo test.
5. Prima di diffondere i numeri: fai leggere il significato clinico a un cardiologo; ricorda che SHD (cardiopatia strutturale) non coincide con scompenso cardiaco.

## 18. Cosa non è stato verificato (pipeline 2)

- Tutte le fasi Q0-Q10, la ripresa dopo errori, la pausa per l'approvazione, il secondo test (con impronte e uso singolo) e il reset sono state collaudate con un finto `claude` e con dati sintetici. Il motore della CNN (`tools/cnn_cv.py`: cross-validation, ripresa per job, riepilogo con intervalli per paziente, candidati) è stato provato su dati sintetici **su CPU**, non su GPU e non sui dati reali.
- Con un Claude vero sono state provate solo le prime fasi su dati sintetici (vedi `BUILD_PROGRESS.md` per l'elenco aggiornato): **i prompt delle altre fasi vanno verificati alla prima esecuzione vera** e probabilmente rifiniti.
- Costi e tempi sui dati reali non sono stimati: li misura Q0 (prova dei tempi) e li stampa il report.
- L'ordine delle derivazioni ECG non è documentato da PhysioNet: Q0 lo verifica con le relazioni tra derivazioni; se il verdetto non è `yes`, i report parlano di «canali» e non di derivazioni con nome.
- Le soglie (classi di accuratezza, credibilità di sottogruppi, criteri di conferma al test) sono valori ragionevoli, non standard: discutile con i colleghi.
- Gli agenti sono modelli di linguaggio: possono sbagliare. Per questo ci sono il revisore indipendente (Q7), i controlli nel codice e le impronte dei candidati. **La lettura finale spetta a una persona** e, per il significato clinico, a un cardiologo.
- Nessuna analisi può dire come il modello si comporterebbe in altri ospedali: serve una validazione esterna, che richiederebbe dati che oggi non sono disponibili.

## 19. Mappa dei file nuovi

| File | A cosa serve |
|---|---|
| `lib/pipe2_lib.py` | Pool train+validation, fold per paziente, lettura forme d'onda, bootstrap per paziente, calibrazione, Holm |
| `tools/cnn_cv.py` | CNN in cross-validation riprendibile, riepilogo, addestramento finale, candidati |
| `tools/state2.py`, `tools/phases2.py` | Stato e definizione delle fasi Q0-Q10 (file `agentic/pipeline2_state.json`, `DECISIONS2.md`, `FROZEN2.json`) |
| `tools/run_pipeline2.py` | Driver della pipeline 2 (log in `agentic/RUNLOG2.md`) |
| `tools/final_eval2.py` | Secondo test: uso singolo, controllo impronte |
| `tools/reset2.py` | Azzera solo la pipeline 2 |
| `agentic/PLAYBOOK2.md` | Istruzioni di ogni fase e regole di decisione |
| `.claude/agents/*` (11 nuovi) | Un agente per fase Q0-Q10 |
| `.claude/commands/run-pipeline2.md` | Comando interattivo `/run-pipeline2` |
| `GUIDA_COMPLETA.md` | Questa guida (Parte 1 invariata + Parte 2) |
