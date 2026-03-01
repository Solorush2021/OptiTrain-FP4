import os
import sys
import json
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Define a standard search space for NVFP4 and Muon Optimizer hyperparameters
SEARCH_SPACE = {
    "lr": [1e-3, 2e-3, 5e-3],
    "momentum_lr": [1e-4, 2e-4],
    "beta": [0.95, 0.98],
    "qk_max_ratio": [0.01, 0.02, 0.05],
    "micro_block_size": [16, 32],
    "stochastic_rounding": [True, False]
}

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=optitrain_fp4_sweep_{job_id}
#SBATCH --output=logs/sweep_{job_id}.out
#SBATCH --error=logs/sweep_{job_id}.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --partition=blackwell_short

module load cuda/12.4
module load pytorch/2.3.0

# DeepSpeed multi-node/multi-gpu training launch command
deepspeed --num_gpus=8 train_sweep.py \\
    --lr {lr} \\
    --momentum_lr {momentum_lr} \\
    --beta {beta} \\
    --qk_max_ratio {qk_max_ratio} \\
    --micro_block_size {micro_block_size} \\
    --stochastic_rounding {stochastic_rounding} \\
    --output_dir results/{job_id}
"""

class AutoResearcher:
    """
    Karpathy-style auto-research coordinator. Generates sweeps, schedules Slurm jobs, 
    tracks outcomes, and ranks configurations based on convergence and training stability.
    """
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / "jobs"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Cartesian product of search space to create individual runs."""
        keys, values = zip(*SEARCH_SPACE.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return experiments

    def create_job_script(self, job_idx: int, params: Dict[str, Any]) -> Path:
        """Generates the SBATCH script for a given parameter set."""
        job_id = f"run_{job_idx:04d}"
        script_content = SLURM_TEMPLATE.format(
            job_id=job_id,
            lr=params["lr"],
            momentum_lr=params["momentum_lr"],
            beta=params["beta"],
            qk_max_ratio=params["qk_max_ratio"],
            micro_block_size=params["micro_block_size"],
            stochastic_rounding=str(params["stochastic_rounding"]).lower()
        )
        
        script_path = self.jobs_dir / f"submit_{job_id}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path

    def run_sweep(self, dry_run: bool = True):
        """Orchestrates the sweeps and submits jobs."""
        experiments = self.generate_experiments()
        print(f"[*] Generated {len(experiments)} hyperparameter configurations for sweeping.")
        
        # Save search catalog to disk
        catalog_path = self.base_dir / "sweep_catalog.json"
        with open(catalog_path, "w") as f:
            json.dump({f"run_{i:04d}": params for i, params in enumerate(experiments)}, f, indent=4)
        
        for idx, exp in enumerate(experiments):
            script_path = self.create_job_script(idx, exp)
            if not dry_run:
                print(f"[+] Submitting Slurm job for run_{idx:04d}...")
                subprocess.run(["sbatch", str(script_path)])
            else:
                print(f"[Dry Run] Generated script at {script_path.relative_to(self.base_dir)}")
                
    def analyze_results(self) -> Dict[str, Any]:
        """
        Parses logs and results to rank experiments by validation loss 
        and check if self-attention QK collapse or divergence occurred.
        """
        summary = {}
        for result_file in self.results_dir.glob("**/stats.json"):
            run_id = result_file.parent.name
            try:
                with open(result_file, "r") as f:
                    stats = json.load(f)
                    
                # Metrics of interest
                val_loss = stats.get("val_loss", float("inf"))
                qk_entropy = stats.get("qk_entropy", 0.0)
                diverged = stats.get("diverged", False)
                
                summary[run_id] = {
                    "val_loss": val_loss,
                    "qk_entropy": qk_entropy,
                    "diverged": diverged,
                    "step_count": stats.get("step_count", 0)
                }
            except Exception as e:
                print(f"[-] Error reading result for {run_id}: {e}")
                
        # Rank by validation loss (only if not diverged)
        valid_runs = {k: v for k, v in summary.items() if not v["diverged"]}
        sorted_runs = sorted(valid_runs.items(), key=lambda item: item[1]["val_loss"])
        
        analysis = {
            "best_run": sorted_runs[0] if sorted_runs else None,
            "total_runs": len(summary),
            "diverged_count": len(summary) - len(valid_runs),
            "ranking": sorted_runs
        }
        
        # Write report to markdown
        report_path = self.base_dir / "research_findings.md"
        with open(report_path, "w") as f:
            f.write("# Hyperparameter Sweep & Stability Analysis Report\n\n")
            f.write(f"- **Total runs evaluated**: {analysis['total_runs']}\n")
            f.write(f"- **Diverged / Entropy Collapsed runs**: {analysis['diverged_count']}\n\n")
            
            if sorted_runs:
                f.write("## Top 5 Configurations\n\n")
                f.write("| Rank | Run ID | Val Loss | QK Attention Entropy | Status |\n")
                f.write("|---|---|---|---|---|\n")
                for rank, (run_id, data) in enumerate(sorted_runs[:5], 1):
                    f.write(f"| {rank} | {run_id} | {data['val_loss']:.4f} | {data['qk_entropy']:.4f} | Stable |\n")
            else:
                f.write("No stable runs found. Consider lower learning rates or adjusting MuonClip thresholds.\n")
                
        return analysis

if __name__ == "__main__":
    researcher = AutoResearcher()
    # If run with --submit, actually schedule on cluster; else do dry_run
    submit_jobs = len(sys.argv) > 1 and sys.argv[1] == "--submit"
    researcher.run_sweep(dry_run=not submit_jobs)
