"""Resolve project-relative references in historical, checksum-protected manifests."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
CNN_CHECKPOINTS = OUTPUTS / "RQ1/baselines/cnn/checkpoints"
RF_CHECKPOINTS = OUTPUTS / "RQ1/baselines/random_forest/checkpoints"


@lru_cache(maxsize=1)
def _old_outputs():
    return json.loads((OUTPUTS / "records/output_layout.json").read_text())["old_to_new"]


def recorded_hash_path(name):
    """Read original documentation when verifying a snapshot taken before the move."""
    if str(name) == "dataset/数据处理与任务定义.md":
        return OUTPUTS / "records/pre_reorganization/project" / name
    target = recorded_path(name)
    if not target.exists() and target.is_relative_to(OUTPUTS / "vlm"):
        archived = OUTPUTS / "records/vlm_training" / target.relative_to(OUTPUTS / "vlm")
        if archived.is_file():
            return archived
    if not target.exists() and target.is_relative_to(OUTPUTS):
        manifest = OUTPUTS / "records/cleanup_manifest.json"
        if manifest.is_file():
            archive = Path(json.loads(manifest.read_text())["archive_root"])
            relative = target.relative_to(ROOT)
            candidate = archive / relative
            if candidate.is_file():
                return candidate
    return target


def recorded_path(name):
    path = Path(name)
    parts = path.parts
    if parts[:2] in {("outputs", "baselines"), ("outputs", "foundation")}:
        return OUTPUTS.joinpath("RQ1", *parts[1:])
    if parts[:1] == ("dataset",):
        return ROOT.joinpath("data", *parts[1:])
    if parts[:1] == ("models",) and len(parts) == 2:
        if parts[1] in {"cnn.pt", "cnn_best.pt"}:
            return CNN_CHECKPOINTS / parts[1]
        if parts[1] == "random_forest.joblib":
            return RF_CHECKPOINTS / parts[1]
    if parts[:1] == ("results",) and len(parts) > 1:
        original = OUTPUTS.joinpath("records/pre_reorganization", *parts[1:])
        if original.is_file():
            return original
        mapping = _old_outputs()
        if parts[1] == "models" and len(parts) == 3:
            return OUTPUTS / mapping[str(Path(*parts[1:]))]
        return OUTPUTS.joinpath(mapping[parts[1]], *parts[2:])
    return ROOT / path
