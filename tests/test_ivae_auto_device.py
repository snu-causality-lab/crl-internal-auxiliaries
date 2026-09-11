"""Exercise real iVAE device placement without training; require ML dependencies."""

import importlib.util
from pathlib import Path
import random
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.reproduce_ivae_auto_device import run_probe

HAS_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "pytorch_lightning", "numpy")
)


@unittest.skipUnless(HAS_DEPENDENCIES, "requires the repository's PyTorch/Lightning environment")
class IVAEDeviceTests(unittest.TestCase):
    def test_cpu_control(self):
        report = run_probe("cpu")
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["constructor_device"], "cpu")
        self.assertFalse(report["accelerator_exercised"])

    def test_auto_on_actual_accelerator(self):
        import torch

        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            self.skipTest("no CUDA/MPS accelerator; CPU alone cannot test this regression")
        report = run_probe("auto")
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["accelerator_exercised"])
        self.assertEqual(report["sampling_device"], report["parameter_device"])
        self.assertEqual(report["decoder_variance_device"], report["parameter_device"])

    def test_explicit_accelerator_control(self):
        import torch

        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            self.skipTest("no CUDA/MPS accelerator for the explicit-device control")
        report = run_probe(accelerator)
        self.assertEqual(report["status"], "passed", report)
        self.assertTrue(report["accelerator_exercised"])

    def test_trainer_construction_preserves_random_states(self):
        import numpy as np
        import pytorch_lightning as pl
        import torch

        for accelerator in ("cpu", "auto"):
            with self.subTest(accelerator=accelerator):
                python_before = random.getstate()
                numpy_before = np.random.get_state()
                cpu_before = torch.get_rng_state().clone()
                cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                mps_before = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
                pl.Trainer(
                    max_epochs=1, logger=None, callbacks=[],
                    check_val_every_n_epoch=1, accelerator=accelerator, devices=1,
                )
                self.assertEqual(python_before, random.getstate())
                numpy_after = np.random.get_state()
                self.assertEqual(numpy_before[0], numpy_after[0])
                np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
                self.assertEqual(numpy_before[2:], numpy_after[2:])
                self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
                for old, new in zip(cuda_before, torch.cuda.get_rng_state_all()):
                    self.assertTrue(torch.equal(old, new))
                if mps_before is not None:
                    self.assertTrue(torch.equal(mps_before, torch.mps.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
