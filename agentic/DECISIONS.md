# Registro decisioni (append-only)

| Quando | Tipo | Fase | Testo |
|---|---|---|---|
| 2026-11-06 12:05:01 | config | - | pipeline inizializzata, data_dir=/home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1 |
| 2026-11-06 12:07:16 | issue | P0 | File .npy delle forme d'onda TRONCATI: header coerente col CSV ma byte reali coprono solo ~25% (train 18299/72475 righe), 40% (val 1846/4626), 68% (test 3685/5442); mmap fallisce. Riscaricare da PhysioNet. P4 bloccata; fasi tabulari OK. |
| 2026-11-06 12:07:54 | issue | P0 | P0 STATUS PARZIALE: integrità OK (overlap pazienti 0/0/0) ma forme d'onda .npy troncate (righe complete train 18299/72475, val 1846/4626, test 3685/5442). P4 bloccata finché non vengono riscaricati i file e rieseguita P0. Shift prevalenza train 0.5237 vs val 0.4302; pulmonary_regurgitation 21 positivi in val (<25); pr_interval mancante 10.5%, lvef_value 8.9%/11.1%. |
| 2026-11-06 12:07:54 | decision | P0 | Nessuna modifica a config e nessun follow-up (0/4 usati). Il PARZIALE riguarda solo le forme d'onda; integrità OK e GPU=sì (nessuna assunzione GPU necessaria), quindi sufficiente per proseguire con le fasi tabulari P1, P2, P3, P5. P4 non va lanciata finché i .npy non sono riscaricati; il driver deve fermarsi su P4. |
| 2026-11-06 12:09:01 | decision | P1 | TABULAR_SIGNAL yes: gain 0.0877>=0.02, IC inf 0.733>0.55; CV train 0.728 vs val 0.746 coerenti |
