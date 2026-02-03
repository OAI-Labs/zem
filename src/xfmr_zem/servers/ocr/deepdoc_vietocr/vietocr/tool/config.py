import yaml

def load_config(config_file):
    with open(config_file, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname, base_file=None):
        from pathlib import Path
        if base_file is None:
            base_file = Path(__file__).resolve().parent.parent / 'config' / 'base.yml'

        base_config = load_config(base_file)

        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        from pathlib import Path
        config_dir = Path(__file__).resolve().parent.parent / 'config'
        return Cfg.load_config_from_file(config_dir / f'{name}.yml')


    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)
