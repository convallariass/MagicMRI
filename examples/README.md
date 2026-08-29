# Synthetic smoke-test fixtures

The repository contains 18 deterministic software fixtures: six translation, six enhancement/restoration, and six segmentation cases. Each directory provides an exemplar source, an exemplar target, a query source, and task metadata. The exemplar source-target relationship defines the requested operation; the query source is the input on which that operation is requested.

These synthetic fixtures are provided only for software smoke testing and are not samples from the study cohorts and are not used to reproduce paper-level quantitative results.

They test package imports, checkpoint loading, prompt construction, each inference-family path, output shape/non-emptiness, and binary segmentation rendering. They are not representative MRI examples, modality-faithful examples, clinically meaningful examples, or paper evidence.
