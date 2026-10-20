# Inference assets

This directory contains the six selected classifiers and the original configuration / feature extraction code required to run them. `manifest.json` records file hashes and model metadata. Each `feature_schema.json` records the exact ordered input columns and their schema fingerprint.

No training rows, source datasets, ablation outputs, or downloaded CodeBERTa weights are included. Original experiment directory names are preserved so the adapters can also load a local `emb-_cbt/` tree. The application supports both packaged JSON schemas and original CSV headers.

Models require `scikit-learn==1.8.0`. Prepare the encoder with `python scripts/prepare_models.py` after installing dependencies. See the root [English README](../README.md) or [中文说明](../README.zh-CN.md).

To export assets from an original research checkout:

```powershell
python scripts/package_assets.py --source emb-_cbt --target exported-models
```

The exporter validates source assets before writing and refuses to overwrite an existing destination. It copies model and extractor bytes unchanged and writes only feature column names into the schema.
