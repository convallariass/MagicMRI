import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.aggregation import family_summaries, patient_summaries, task_summaries
from evaluation.metrics import generation_metrics, segmentation_metrics
from magicmri.infer import render_prediction
from magicmri.inference import infer


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def forward_encoder(self, source, target, masked):
        self.shape = source.shape
        return source

    def forward_decoder(self, latent):
        return torch.zeros(self.shape, dtype=torch.float32)


class MetricAndInterfaceTests(unittest.TestCase):
    def test_inference_fixture_smoke(self):
        example = ROOT / "examples/translation/T01"
        output = infer(
            FakeModel(),
            {
                "exemplar_source": example / "exemplar_source.png",
                "exemplar_target": example / "exemplar_target.png",
                "query_source": example / "query_source.png",
            },
            torch.device("cpu"),
            input_size=32,
        )
        self.assertEqual(output.shape, (32, 32, 3))
        self.assertGreater(output.size, 0)

    def test_metric_smoke(self):
        target = np.zeros((16, 16, 3), dtype=np.uint8)
        prediction = target.copy()
        image_metrics = generation_metrics(prediction, target)
        self.assertEqual(image_metrics, {"SSIM": 1.0, "PSNR": 100.0, "NMAE": 0.0})
        mask_metrics = segmentation_metrics(prediction[..., 0], target[..., 0])
        self.assertEqual(mask_metrics["Dice"], 1.0)
        self.assertEqual(mask_metrics["mIoU"], 0.5)
        self.assertEqual(mask_metrics["pACC"], 1.0)
        self.assertEqual(mask_metrics["HD95"], 0.0)

    def test_segmentation_binary_output_smoke(self):
        prediction = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        rendered = render_prediction(prediction, "binary_mask", 128)
        self.assertEqual(set(np.unique(rendered)), {0, 255})

    def test_aggregation_order(self):
        rows = []
        for patient, value in (("p1", 0.0), ("p1", 1.0), ("p2", 1.0)):
            rows.append({
                "task_id": "task", "family": "translation", "metric_family": "generation",
                "patient_id": patient, "SSIM": value, "PSNR": value, "NMAE": value, "LPIPS": value,
            })
        patients = patient_summaries(rows)
        tasks = task_summaries(rows)
        families = family_summaries(tasks)
        self.assertEqual(len(patients), 2)
        self.assertAlmostEqual(tasks[0]["SSIM"], 2.0 / 3.0)
        self.assertAlmostEqual(families[0]["SSIM"], 2.0 / 3.0)

    def test_formal_evaluator_subset_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.png"
            target = root / "target.png"
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            Image.fromarray(mask).save(prediction)
            Image.fromarray(mask).save(target)
            manifest = root / "predictions.jsonl"
            manifest.write_text(json.dumps({
                "sample_id": "sample", "task_id": "tumor_segmentation-t1c_Glioma",
                "patient_id": "patient", "slice_id": "slice", "slice_index": 0,
                "exemplar_source": str(prediction), "exemplar_target": str(target),
                "query_source": str(prediction), "query_target": str(target),
                "prediction_path": str(prediction),
            }) + "\n", encoding="utf-8")
            output = root / "metrics"
            result = subprocess.run(
                [sys.executable, "evaluation/core36_evaluator.py", "--manifest", str(manifest),
                 "--task-registry", "configs/core36_tasks.yaml", "--output-dir", str(output),
                 "--device", "cpu", "--skip-lpips", "--allow-task-subset"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "per_slice.jsonl").is_file())
            summary = json.loads((output / "task_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["Dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
