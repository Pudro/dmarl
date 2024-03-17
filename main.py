import argparse
import yaml
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser("MAgent2")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config['env']['config_file'] = f.name
        _validate_config(config)
        # copy env config to each egent
        for agent in config["agents"]:
            agent.update(config["env"])

        return _dict_to_namespace(config)


def _validate_config(config):
    for el in ("env", "agents"):
        try:
            config[el]
        except:
            raise ValueError(f"The config file is missing field: {el}")

    if not isinstance(config["agents"], list):
        raise AttributeError(
            f"Field: agents in config file is not a list. Got config: {config}"
        )

    if not isinstance(config["agents"][0], dict):
        raise AttributeError(
            f"Field: agents in config file is not formatted properly. Got config: {config}"
        )

    if not isinstance(config["env"], dict):
        raise AttributeError(
            f"Field: env in config file is not formatted properly. Got config: {config}"
        )


def _dict_to_namespace(config):
    for key in config.keys():
        if key == "agents":
            for i, _ in enumerate(config[key]):
                config[key][i] = argparse.Namespace(**config[key][i])
        else:
            config[key] = argparse.Namespace(**config[key])

    config = argparse.Namespace(**config)
    return config


if __name__ == "__main__":
    config = parse_args()
    runner = Runner(config)
