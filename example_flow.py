import subprocess
import os
import sys
seeds = list(range(0,20))

result_dir = "./result_ours_flow"
dgp_name = "c_real"
if __name__ == "__main__":
    os.makedirs(result_dir, exist_ok=True)
    dgp_dir = os.path.join(result_dir, dgp_name)
    os.makedirs(dgp_dir, exist_ok=True)
    for seed in seeds:
        command = [
            sys.executable,
            "-m",
            "experiments.Ours.main",
            f"--dgp={dgp_name}",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--lr-scheduler=cosine",
            "--mode=flow",
            "--max-epochs=50",
            "--check-val-every-n-epoch=50",
            "--lr=0.001",
            "--k-flows=8",
        ]
        subprocess.run(command, check=True)
