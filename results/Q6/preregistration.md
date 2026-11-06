# Preregistrazione Q6 (scritta il 2026-11-06, prima di qualsiasi addestramento e senza guardare il test)

Test set chiuso: nessuna scelta usa il test. Dati usati per scegliere: solo gli Handoff di Q1-Q5 (CV per paziente su train+validation, popolazione clean).

## Regole di scelta
- C1: CNN con CONFIG_FINALE di Q1 (epochs 12, lr 0.002, reg_weight 0.5, batch 128, 12 derivazioni, popolazione clean), uscita SHD. Sempre.
- C2: stessa rete, uscita LVEF <= 45%. Sempre.
- C3..: altre etichette in TARGETS_GOOD di Q1 con >= 100 positivi in clean, in ordine di AUROC CV: rv_systolic_dysfunction (0.844, 9965 pos), aortic_stenosis (0.811, 3171 pos).
- Un candidato in più con INSIEME_MINIMO_SUFFICIENTE di Q2 (3 derivazioni, I_II_V1), uscita SHD.
- Massimo 6 candidati (config pipeline 2): totale previsto 5.
- Nessun candidato da Q3, Q4, Q5 (nessun sottogruppo/cluster credibile). Non si usano strati per scegliere.
- Pesi: final-fit con 3 seed (0,1,2) su tutto il pool di sviluppo clean; la predizione è la media dei seed.
- Comparatori tabulari: results/P1/candidates/global_best (SHD), results/P3/candidates/lvef_lte45_global (LVEF<=45%); nessuno per gli altri.

## Criteri di conferma al test (fissati ora)
- "Confermato": AUROC test con IC95% (per paziente) inferiore >= 0.75 E calo rispetto all'AUROC CV clean <= 0.05.
- "Buona": AUROC test >= 0.80 con IC inferiore >= 0.75. Moderata 0.70-0.80, debole < 0.70.
- AUROC CV di riferimento: C1 0.815, C2 0.869, C3 0.844, C4 0.811 (ensemble 3 seed, Q1); C5 0.802 (solo seed 0, Q2).
- Nota: per la stenosi aortica il tabulare era migliore della CNN in CV (0.841 vs 0.811): atteso che la CNN non batta il comparatore.
