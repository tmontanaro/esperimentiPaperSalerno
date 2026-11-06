# Registro decisioni (append-only)

| Quando | Tipo | Fase | Testo |
|---|---|---|---|
| 2026-11-06 17:36:58 | config | - | pipeline 2 inizializzata, data_dir=/home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1 |
| 2026-11-06 17:40:07 | decision | Q0 | LEAD_ORDER_OK=yes (R2 0.976-0.996, segni attesi); tempi ~9.5 s/epoca GPU; Q1 ~30 min, Q2 <=70 min, Q6 ~8 min |
| 2026-11-06 19:58:23 | decision | Q2 | INSIEME_MINIMO_SUFFICIENTE = 3 derivazioni (I, II + una toracica): diff SHD -0.006, LVEF<=45 -0.008 vs all |
| 2026-11-06 20:01:48 | decision | Q3 | QRS_MODIFICA=non_supportata: dAUROC +0.0001 [-0.0005,+0.0006], dR2 +0.0004 [-0.0002,+0.0010]; 8 confronti, Holm per famiglia |
| 2026-11-06 20:24:42 | decision | Q4 | Nessun cluster credibile sugli embedding (k=3 in ogni fold, ARI 0.90-0.92); strati: la CNN peggiora per QRS<100, PR corto, QTc basso, ambulatoriale (diff fino a -0.05); 90 strati e 32 regole, Holm |
| 2026-11-06 20:33:31 | decision | Q6 | Preregistrazione scritta: 5 candidati (SHD, LVEF<=45, RV, stenosi aortica, SHD con 3 derivazioni); criteri conferma IC inf>=0.75 e calo<=0.05 |
| 2026-11-06 20:45:03 | decision | Q6 | congelati 5 candidati per il secondo test (impronte sha256 registrate) |
