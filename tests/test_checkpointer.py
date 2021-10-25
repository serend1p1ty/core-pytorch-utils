import torch
import torch.nn as nn

from cpu.checkpoint import Checkpointer


def create_model():
    model = nn.Module()
    model.layer1 = nn.Linear(2, 3)
    model.layer2 = nn.Linear(3, 2)
    model.res = nn.Module()
    model.res.layer2 = nn.Linear(3, 2)

    model_state_dict = {}
    model_state_dict["layer1.weight"] = torch.rand(3, 2)
    model_state_dict["layer1.bias"] = torch.rand(3)
    model_state_dict["layer2.weight"] = torch.rand(2, 3)
    model_state_dict["layer2.bias"] = torch.rand(2)
    model_state_dict["res.layer2.weight"] = torch.rand(2, 3)
    model_state_dict["res.layer2.bias"] = torch.rand(2)
    return model, {"model": model_state_dict}


def test_checkpointer():
    model, state_dict = create_model()
    model_sd = model.state_dict()
    checkpointer = Checkpointer("", model=model)
    checkpointer.load(checkpoint=state_dict)
    for loaded, stored in zip(model_sd.values(), state_dict.values()):
        # different tensor references
        assert id(loaded) != id(stored)
        # same content
        assert torch.all(loaded == stored)
