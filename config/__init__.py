import os

import yaml

config = {}

path = os.path.join(os.path.dirname(__file__), "config.yaml")
if os.path.exists(path):
    with open(path, "r") as stream:
        config = {**config, **yaml.safe_load(stream)}

for key in config.keys():
    if key in os.environ:
        config[key] = yaml.safe_load(os.environ.get(key))
