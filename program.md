# Objective: Optimize NVFP4 Pretraining Convergence & Stability with Muon Optimizer

Optimize the hyperparameters of the 4-bit mixed-precision pretraining simulation to minimize validation MSE loss on our target block-tiling architecture, subject to constraints.

## Target File
- Code to edit: `train_sweep.py` (specifically hyperparameter declarations) or optimizer configuration.

## Metrics of Success
1. **Primary Metric**: Validation Mean Squared Error (MSE) loss (minimize, baseline target: `< 1.90`, initial config gives `2.02`).
2. **Constraint Metric**: Attention Entropy (must remain `> 0.1` at all times during the 50-step run).
3. **Execution Limit**: Pilot training runs are capped at 50 steps or 3 minutes.

## Search Space Parameters
- Learning Rate ($\eta$): Range `[1e-4, 5e-3]`
- Momentum Learning Rate ($\eta_{mom}$): Range `[1e-5, 5e-4]`
- QK-Clip Ratio ($\alpha_{limit}$): Range `[0.01, 0.05]`
- Newton-Schulz Orthogonalization steps ($K$): Range `[3, 8]`

## Instructions for the AI Agent
1. **Propose**: Tweak parameters or architecture logic in `train_sweep.py` or `optimizer.py`.
2. **Execute**: Launch the training pilot via `python train_sweep.py --lr <lr> --momentum_lr <momentum_lr> --qk_max_ratio <qk_max_ratio>`.
3. **Evaluate**: Check validation loss and attention entropy in the output JSON.
4. **Ratchet**: If loss is reduced and attention entropy remains stable (> 0.1), perform a Git commit to lock in the improvements. Otherwise, revert the changes via Git.
