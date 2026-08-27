#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magicmri.models.magicmri import MagicMRI  # noqa: E402
from magicmri.data.dataset import VisualPairDataset  # noqa: E402


class ReleaseSmokeTests(unittest.TestCase):
    def test_examples_are_complete(self):
        rows = json.loads((ROOT / "examples" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 18)
        counts = {family: 0 for family in ("translation", "enhancement", "segmentation")}
        for row in rows:
            counts[row["family"]] += 1
            directory = ROOT / "examples" / row["directory"]
            config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            self.assertTrue(config["synthetic"])
            for name in ("exemplar_source.png", "exemplar_target.png", "query_source.png"):
                self.assertTrue((directory / name).is_file())
        self.assertEqual(counts, {"translation": 6, "enhancement": 6, "segmentation": 6})

    def test_tiny_model_forward_shape(self):
        model = MagicMRI(
            img_size=(32, 16),
            patch_size=16,
            embed_dim=32,
            depth=24,
            num_heads=4,
            mlp_ratio=2,
            drop_path_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            decoder_embed_dim=8,
            pretrain_img_size=16,
        )
        images = torch.randn(1, 3, 32, 16)
        targets = torch.randn(1, 3, 32, 16)
        mask = torch.tensor([[0, 1]], dtype=torch.bool)
        loss, prediction, observed_mask = model(images, targets, mask, torch.ones_like(targets))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(prediction.shape), (1, 2, 16 * 16 * 3))
        self.assertEqual(tuple(observed_mask.shape), (1, 2))

    def test_visual_pair_dataset_shapes(self):
        example = ROOT / "examples" / "translation" / "T01"
        records = [
            {
                "image_path": str(example / "query_source.png"),
                "target_path": str(example / "exemplar_target.png"),
                "type": "translation-smoke",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.json"
            manifest.write_text(json.dumps(records), encoding="utf-8")
            dataset = VisualPairDataset(
                root="/",
                manifests=[manifest],
                image_size=32,
                training=False,
                half_mask_ratio=1.0,
                num_mask_patches=4,
                max_mask_patches_per_block=4,
            )
            source, target, mask, valid = dataset[0]
        self.assertEqual(tuple(source.shape), (3, 64, 32))
        self.assertEqual(tuple(target.shape), (3, 64, 32))
        self.assertEqual(tuple(mask.shape), (4, 2))
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(tuple(valid.shape), (3, 64, 32))


if __name__ == "__main__":
    unittest.main()
