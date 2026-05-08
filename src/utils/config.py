from pathlib import Path
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(cfg):
    output_cfg = cfg["output"]
    for key in [
        "checkpoint_dir",
        "figure_dir",
        "table_dir",
        "log_dir",
        "sample_dir",
    ]:
        Path(output_cfg[key]).mkdir(parents=True, exist_ok=True)
