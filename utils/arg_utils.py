# -*- coding:UTF-8 -*-
"""
Fetching arguments in args.yaml
@Cai Yichao 2020_09_08
"""

import yaml


def fetch_args():
    yaml_path = 'args.yml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        args = f.read()
        args = yaml.load(args)
        return args


