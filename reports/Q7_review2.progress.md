# Q7 review2 · progress
- [x] brief, handoff Q0-Q6, PLAYBOOK2 §Q7 letti
- [x] 1. fold per paziente: experiments/Q7/check_folds_auc.py -> results/Q7/check_folds_auc.json, fold_checks.csv (56 npz, 10 run, 0 sovrapposizioni)
- [x] 2. riproduzione AUROC con sklearn: 216 confronti, diff max 0.000 (results/Q7/auc_reproduction.csv)
- [x] 3. popolazioni clean: n_train corretto in 56/56 npz
- [x] 4. test chiuso: experiments/Q7/check_test_frozen.py (final2 assente, nessun riferimento al test negli script Q*)
- [x] 5. multiplicità / Holm: experiments/Q7/check_q3_q5.py (Holm Q3/Q4 identica)
- [x] 6. tabella affermazioni Q1-Q5 (nel report)
- [x] 7. candidati: sha256 30/30, contratto experiments/Q7/check_contract.py, regole e tempi OK
- [x] extra: occlusione canale (experiments/Q7/check_occlusion.py)
- [x] report reports/Q7_review2.md + validate
Esito: PASS, nessun candidato rimosso.
