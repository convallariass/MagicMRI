import os
import tempfile
import unittest
from pathlib import Path

import torch

from magicmri.tasks import load_task_registry
from magicmri.utils.checkpoint import load_checkpoint, load_pretrained_model


ROOT = Path(__file__).resolve().parents[1]


class Core36RegistryTests(unittest.TestCase):
    def test_exact_core36_registry(self):
        tasks = load_task_registry(ROOT / "configs/core36_tasks.yaml")
        self.assertEqual(len(tasks), 36)
        self.assertEqual(len({task["task_id"] for task in tasks}), 36)
        self.assertEqual(sum(task["family"] == "translation" for task in tasks), 12)
        self.assertEqual(sum(task["family"] == "enhancement" for task in tasks), 12)
        self.assertEqual(sum(task["family"] == "segmentation" for task in tasks), 12)
        translation = {
            (task["source_modality"], task["target_modality"])
            for task in tasks
            if task["family"] == "translation"
        }
        modalities = {"T1n", "T1c", "T2f", "T2w"}
        self.assertEqual(translation, {(source, target) for source in modalities for target in modalities if source != target})
        enhancement = {
            (task["source_modality"], task["degradation"])
            for task in tasks
            if task["family"] == "enhancement"
        }
        self.assertEqual(
            enhancement,
            {(modality, degradation) for modality in modalities for degradation in ("blurx2", "gaussian_noise", "salt_and_pepper_noise")},
        )
        segmentation = {
            (task["source_modality"], task["tumor_target"])
            for task in tasks
            if task["family"] == "segmentation"
        }
        self.assertEqual(
            segmentation,
            {(modality, tumor) for modality in modalities for tumor in ("glioma", "meningioma", "metastasis")},
        )


class CheckpointSchemaTests(unittest.TestCase):
    def test_checkpoint_state_dict_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pth"
            torch.save({"model": {"weight": torch.ones(1)}}, path)
            state = load_checkpoint(path)
        self.assertEqual(set(state), {"weight"})
        self.assertTrue(torch.equal(state["weight"], torch.ones(1)))

    @unittest.skipUnless(os.environ.get("MAGICMRI_CKPT"), "set MAGICMRI_CKPT for release integration")
    def test_release_checkpoint_strict_load_if_configured(self):
        model = load_pretrained_model(os.environ["MAGICMRI_CKPT"], torch.device("cpu"))
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
