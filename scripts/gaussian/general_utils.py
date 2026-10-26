import os
import torch
import pytz
from datetime import datetime
import yaml

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_name(cfg=None):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    if cfg is not None:
        formatted_string = f"{current_month:02d}-{current_day:02d}-{current_hour:02d}-{current_minute:02d}-{cfg['dataset']['module'].split('.')[-1]}"
    else:
        formatted_string = f"{current_month:02d}-{current_day:02d}-{current_hour:02d}-{current_minute:02d}"
    return formatted_string

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _localize(cfg):
    """Repoint the author's absolute paths at this checkout.

    Every config ships with paths under /data/wuke/... that only exist on the
    machine the configs were written on. We rewrite them onto this checkout so a
    fresh clone works with no yaml editing:
      /data/wuke/workspace/VINGS-Mono/<x>          ->  <root>/<x>
      /data/wuke/workspace/September/VINGS-Mono/<x>->  <root>/<x>
      /data/wuke/DATA/<x>                          ->  <root>/data/DATA/<x>
      /data/wuke/workspace/<x>                     ->  <root>/data/<x>
    Datasets and third-party weights land under <root>/data/ to keep them out of
    the tracked tree. Override the base with VINGS_ROOT.
    """
    root = os.environ.get('VINGS_ROOT', REPO_ROOT).rstrip('/')
    code_prefixes = ('/data/wuke/workspace/VINGS-Mono/',
                     '/data/wuke/workspace/September/VINGS-Mono/')
    data_prefix = '/data/wuke/'

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            for p in code_prefixes:
                if node.startswith(p):
                    return os.path.join(root, node[len(p):])
            if node.startswith(data_prefix):
                return os.path.join(root, 'data', node[len(data_prefix):])
        return node

    return walk(cfg)


def load_config(cfg_path):
    # Return a Dict.
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.full_load(f)
    return _localize(cfg)