from agents.base import Base_Agent

def get_nn_agent_size(nn_agent: Base_Agent):
    param_size = 0
    for param in nn_agent.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in nn_agent.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size
