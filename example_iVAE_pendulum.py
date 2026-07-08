import subprocess
import os
import sys
# example_list = ['a', 'b_aug', 'b', 'c', 'c_aug', 'c_real']
# example_list = ['c', 'c_aug', 'c_real']
example_list = ['c_real']
seeds = list(range(0,20))
#seeds = [0]
result_dir = "./result_iVAE_pendulum"

if __name__ == "__main__":
    os.makedirs(result_dir, exist_ok=True)
    for seed in seeds:
        for dgp_name in example_list:
            dgp_dir = os.path.join(result_dir, dgp_name)
            os.makedirs(dgp_dir, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "experiments.iVAE.main",
                f"--dgp={dgp_name}",
                f"--seed={seed}",
                f"--training-seed={seed}",
                f"--result-dir={result_dir}",
                "--mode=pendulum",
                "--lr=0.0001",
                "--max-epochs=80",
            ]
            subprocess.run(command, check=True)
