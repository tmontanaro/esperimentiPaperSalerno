# Registro decisioni (append-only)

| Quando | Tipo | Fase | Testo |
|---|---|---|---|
| 2026-11-06 14:14:51 | config | - | pipeline inizializzata, data_dir=/home/idalab/Desktop/Teo/physionet.org/files/echonext/1.1.1 |
| 2026-11-06 14:19:27 | decision | P0 | Gate P0: STATUS OK, INTEGRITA OK, GPU sì -> nessuna azione; P4 può addestrare in pieno |
| 2026-11-06 14:19:27 | issue | P0 | Prevalenza target train 0.524 vs val 0.430: possibile shift train/val, da considerare nella calibrazione/interpretazione |
| 2026-11-06 14:20:59 | decision | P1 | TABULAR_SIGNAL yes: guadagno 0.0877 (0.7464 hist_gb vs 0.6587 mlp), IC inf 0.733>0.55; candidato global_best = hist_gb |
| 2026-11-06 14:21:37 | config | - | tabular_signal = true |
| 2026-11-06 14:21:37 | decision | P1 | Gate P1: TABULAR_SIGNAL yes (guadagno 0.088, IC inf 0.733) -> tabular_signal=true, clusters_mode resta normal |
| 2026-11-06 14:21:37 | issue | P1 | P1: smoke test non registrato con log-exp e progress file non scritto (deviazioni di processo minori, da segnalare al revisore) |
| 2026-11-06 14:26:32 | decision | P2 | Nessun sottogruppo credibile: 15 regole verificate su val (1950 provate in train), nessuna soddisfa i criteri (aortic stenosis: n_pos val 8-18 <25; composito: IC inf < ref). Promettenti: 9 target su 11 condizioni+composito. |
| 2026-11-06 14:27:57 | config | - | cluster_targets = ["shd_moderate_or_greater_flag","aortic_stenosis_moderate_or_greater_flag","mitral_regurgitation_moderate_or_greater_flag"] |
| 2026-11-06 14:27:57 | decision | P2 | Gate P2: cluster_targets = composito + aortic_stenosis (0.844) + mitral_regurgitation (0.770), i 2 TARGET_PROMETTENTI con AUROC val più alta; nessun SOTTOGRUPPO_CREDIBILE |
| 2026-11-06 14:27:57 | issue | P2 | P2: N_REGOLE_PROVATE=1950 (>500) senza replica indipendente; nessun sottogruppo credibile ne è derivato, ma il revisore ne tenga conto. Scelta target/modello per (b)(c) basata sul validation: AUROC lievemente ottimistiche |
| 2026-11-06 14:31:11 | decision | P3 | QRS_MODIFICA inconclusiva: coppie concordi (4/4 MATCH_ON), Wald interazione p<0.001 ma CV dMSE 0.28 [-0.03,0.56] trascurabile, stratificazione dR2 <=0.009; candidato lvef_lte45_global salvato |
| 2026-11-06 14:32:19 | decision | P3 | Gate P3: QRS_MODIFICA inconclusiva (non supportata) -> nessun follow-up P3b. Stratificazione QRS senza beneficio rilevante; interazione statisticamente significativa ma trascurabile in CV |
| 2026-11-06 14:32:19 | issue | P3 | P3 limiti per P6: Wald raggruppato implementato a mano (statsmodels assente); bootstrap coppie per coppia non per paziente (IC stretti); QRS>=120 non distingue BBS/BBD; modello LVEF<=45 scelto sul validation |
| 2026-11-06 14:39:12 | config | - | waveform_dominant = true |
| 2026-11-06 14:39:12 | decision | P4 | Gate P4: GAIN_VS_TABULARE +0.077 [0.063,0.091] >= 0.05 -> waveform_dominant=true. CNN batte il tabulare su tutte le condizioni tranne aortic_stenosis e in tutte le fasce di QRS |
| 2026-11-06 14:39:12 | issue | P4 | P4: un solo seed (nessuna stima di variabilità tra addestramenti); embedding train non salvati (P5 deve rigenerarli da weights.pt/model_def.py se li usa); results/P4/debug non rimosso (solo smoke test) |
| 2026-11-06 15:21:15 | issue | P5 | fase interrotta: riparte dai file parziali in results/ e dal file .progress.md |
