# P4 waveform - progress
- Smoke test OK (debug_n=2000, 2 epoche: loss 0.565->0.345, val AUROC composito 0.677->0.705). GPU RTX 5080, script experiments/P4/train.py.
- Config scelta: ResNet1D, solo forme d'onda (no feature tabulari), 12 epoche, batch 128, lr 2e-3 OneCycle, epoca FINALE (nessuna selezione su val).
- Run completo: `python3 experiments/P4/train.py --epochs 12` (riprende da results/P4/ckpt.pt)
## Epoche

- [14:36] epoca 1/12 loss=0.3400 val AUROC composito=0.8017 (6s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:36] epoca 2/12 loss=0.2842 val AUROC composito=0.8027 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 3/12 loss=0.2747 val AUROC composito=0.8226 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 4/12 loss=0.2694 val AUROC composito=0.8225 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 5/12 loss=0.2648 val AUROC composito=0.8165 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 6/12 loss=0.2599 val AUROC composito=0.8264 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 7/12 loss=0.2546 val AUROC composito=0.8278 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 8/12 loss=0.2485 val AUROC composito=0.8307 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 9/12 loss=0.2406 val AUROC composito=0.8293 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 10/12 loss=0.2304 val AUROC composito=0.8295 (5s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 11/12 loss=0.2187 val AUROC composito=0.8251 (5s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)
- [14:37] epoca 12/12 loss=0.2112 val AUROC composito=0.8232 (4s) (checkpoint /home/idalab/Desktop/Teo/esperimentiClaude/echonext-agentic/results/P4/ckpt.pt)