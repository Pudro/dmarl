from agents.base import Base_Agent
from agents.random import Random_Agent
from trainers.base import Base_Trainer
from trainers.random import Random_Trainer


def get_nn_agent_size(nn_agent: Base_Agent):
    param_size = 0
    buffer_size = 0

    print(type(nn_agent))
    if not isinstance(nn_agent, Random_Agent):
        for param in nn_agent.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in nn_agent.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size


def count_nn_agent_parameters(nn_agent: Base_Agent):
    if not isinstance(nn_agent, Random_Agent):
        return sum(p.numel() for p in nn_agent.parameters())


def print_summary(trainer_list: list[Base_Trainer]):
    headers = [
        "Side name",
        "Agent count",
        "Agent type",
        "Layer type",
        "Hidden layers",
        "Agent parameter count",
        "Agent size",
        "Test mode"
    ]
    data = []
    for trainer in trainer_list:
        data.append(
            [
                trainer.side_name,
                len(trainer.nn_agents),
                trainer.algorithm,
                trainer.agent_config.layer_type,
                trainer.agent_config.hidden_layers,
                count_nn_agent_parameters(trainer.nn_agents[0]),
                f"{get_nn_agent_size(trainer.nn_agents[0])/1024} KB",
                trainer.agent_config.test_mode,
            ]
        )

    column_widths = [
        max(len(str(item)) + 2 for item in column) for column in zip(headers, *data)
    ]

    header_row = [str(item).center(width) for item, width in zip(headers, column_widths)]
    print("|".join(header_row))
    separators = ['=' * width for width in column_widths]
    print('+'.join(separators))

    for row in data:
        formatted_row = [str(item).center(width) for item, width in zip(row, column_widths)]
        print('|'.join(formatted_row))
        separators = ['-' * width for width in column_widths]
        print('+'.join(separators))
