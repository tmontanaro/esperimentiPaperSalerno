"""Definizione delle fasi della PIPELINE 2 (Q0-Q10). Si aggancia ai risultati della pipeline 1 (P0-P9) senza modificarli."""

CONFIG2 = {
    "data_dir": None,
    "primary_target": "shd_moderate_or_greater_flag",
    "max_followups": 2,
    "max_review_rounds": 2,
    "max_attempts": 3,
    "n_folds": 5,
    "seeds": [0, 1, 2],
    "max_candidates_second_test": 6,
    "thresholds": {   # ereditate dalla pipeline 1 se presente; questi sono i valori di riserva
        "tabular_signal_min_gain": 0.02, "promising_target_auroc_lo": 0.65, "credible_subgroup_margin": 0.02,
        "good_auroc": 0.80, "min_n_val": 100, "min_pos_val": 25,
    },
    "tabular_signal": None, "cluster_targets": [], "clusters_mode": "normal",
}

PHASES = [
    dict(id="Q0", name="inventory", agent="inventory-auditor", depends_on=[], gate=False, priority=0,
         goal="Inventario dei risultati della pipeline 1, verifica dell'ordine delle derivazioni, definizione del protocollo (CV per paziente su train+validation, popolazione «clean»), prova dei tempi."),
    dict(id="Q1", name="cnn-cv", agent="cnn-cv-analyst", depends_on=["Q0"], gate=False, priority=0,
         goal="CNN sulle forme d'onda in cross-validation per paziente, più seed: stima solida e con variabilità di tutte le etichette, confronto col tabulare, sensibilità alle righe senza LVEF."),
    dict(id="Q2", name="lead-ablation", agent="lead-ablation-analyst", depends_on=["Q1"], gate=False, priority=0,
         goal="Rimozione progressiva di derivazioni: quali bastano? (idea originale di togliere input fino a trovare il sottoinsieme che funziona)."),
    dict(id="Q3", name="lvef-qrs-cnn", agent="lvef-qrs-cnn-analyst", depends_on=["Q1"], gate=False, priority=0,
         goal="LVEF con la CNN e ipotesi dei medici sul QRS: la durata del QRS spiega ciò che resta della LVEF dopo la predizione?"),
    dict(id="Q4", name="embedding-subgroups", agent="embedding-analyst", depends_on=["Q1"], gate=False, priority=0,
         goal="Cluster sugli embedding della CNN e sottogruppi in cui la CNN funziona meglio o peggio, con rumore di fondo e correzione per confronti multipli."),
    dict(id="Q5", name="calibration-interpret", agent="calibration-analyst", depends_on=["Q1"], gate=False, priority=0,
         goal="Calibrazione, soglie operative e cosa guarda la rete (occlusione per derivazione e per finestra temporale)."),
    dict(id="Q6", name="candidates", agent="candidate-freezer", depends_on=["Q2", "Q3", "Q4", "Q5"], gate=False, priority=0,
         goal="Scegliere con regole pre-specificate i candidati per il secondo test, addestrarli sul pool dev e congelarli (FROZEN2.json con impronte)."),
    dict(id="Q7", name="review2", agent="skeptic-reviewer2", depends_on=["Q6"], gate=False, priority=0,
         goal="Revisione indipendente della pipeline 2: riproduzione dei numeri, leakage, confronti multipli, test mai toccato, affermazioni non supportate."),
    dict(id="Q8", name="final-test2", agent="final-evaluator2", depends_on=["Q7"], gate=False, priority=0, requires_approval=True,
         goal="Secondo test, pre-specificato, UNA SOLA VOLTA, solo sui candidati congelati; interpretazione onesta."),
    dict(id="Q9", name="synthesis2", agent="synthesizer2", depends_on=["Q8"], gate=False, priority=0,
         goal="Report complessivo in parole povere: obiettivo, prove fatte (pipeline 1 e 2), risultati, cosa è fattibile e cosa no."),
    dict(id="Q10", name="notebook2", agent="notebook-builder2", depends_on=["Q9"], gate=False, priority=0,
         goal="Notebook complessivo che integra quello della pipeline 1 e aggiunge i capitoli della pipeline 2."),
]
