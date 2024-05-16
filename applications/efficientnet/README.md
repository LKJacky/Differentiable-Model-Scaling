# DMS-EfficientNet

## Getting Started

```
pip install -r requirements.txt
```

## Pruning && Retrain

```
export Variant=DMS-450 # for example
scripts/{Variant}/prune.sh
scripts/{Variant}/retrain.sh
```