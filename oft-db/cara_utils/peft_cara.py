from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.bin"
DEFAULT_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


@dataclass
class CaraConfig:
    r: int = 8
    cara_dropout: float = 0.0
    noise_alpha: float = 0.01
    noise_step_interval: int = 5
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    peft_type: str = "CARA"

    def to_dict(self):
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data

    def save_pretrained(self, save_directory: str):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / CONFIG_NAME, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        with open(Path(load_directory) / CONFIG_NAME, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)


class CaraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str = "default",
        r: int = 8,
        cara_dropout: float = 0.0,
        noise_alpha: float = 0.01,
        noise_step_interval: int = 5,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"CaraLinear only supports nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.r = {}
        self.cara_dropout = nn.ModuleDict()
        self.noise_alpha = {}
        self.noise_step_interval = {}
        self.cara_A = nn.ParameterDict()
        self.cara_B = nn.ParameterDict()

        self.active_adapters = [adapter_name]
        self.disable_adapters = False
        self.step_counter = 0

        self.update_layer(
            adapter_name,
            r=r,
            cara_dropout=cara_dropout,
            noise_alpha=noise_alpha,
            noise_step_interval=noise_step_interval,
        )

    def update_layer(self, adapter_name: str, r: int, cara_dropout: float, noise_alpha: float, noise_step_interval: int):
        if r <= 0:
            raise ValueError(f"`r` must be positive, got {r}")

        self.r[adapter_name] = r
        self.cara_dropout[adapter_name] = nn.Dropout(p=cara_dropout) if cara_dropout > 0.0 else nn.Identity()
        self.noise_alpha[adapter_name] = noise_alpha
        self.noise_step_interval[adapter_name] = noise_step_interval

        target_device = self.get_base_layer().weight.device
        target_dtype = torch.float32
        self.cara_A[adapter_name] = nn.Parameter(
            torch.randn(self.in_features, r, device=target_device, dtype=target_dtype) * 0.01
        )
        self.cara_B[adapter_name] = nn.Parameter(
            torch.randn(self.in_features, r, device=target_device, dtype=target_dtype) * 0.01
        )

        self.get_base_layer().weight.requires_grad = False
        if self.get_base_layer().bias is not None:
            self.get_base_layer().bias.requires_grad = False

    def get_base_layer(self) -> nn.Linear:
        return self.base_layer

    def set_adapter(self, adapter_name: str):
        self.active_adapters = [adapter_name]

    def get_input_rotation(self, adapter_name: Optional[str] = None) -> torch.Tensor:
        adapter_name = adapter_name or self.active_adapters[0]
        a_matrix = self.cara_A[adapter_name]
        b_matrix = self.cara_B[adapter_name]
        rank = self.r[adapter_name]

        u_matrix = torch.cat([a_matrix, -b_matrix], dim=1)
        v_matrix = torch.cat([b_matrix, a_matrix], dim=1)
        identity_2r = torch.eye(2 * rank, device=a_matrix.device, dtype=torch.float32)
        inner_matrix = identity_2r + torch.matmul(v_matrix.transpose(0, 1).float(), u_matrix.float())
        z_matrix = torch.inverse(inner_matrix).to(a_matrix.dtype)
        identity = torch.eye(self.in_features, device=a_matrix.device, dtype=a_matrix.dtype)
        return identity - 2 * torch.matmul(torch.matmul(u_matrix, z_matrix), v_matrix.transpose(0, 1))

    def get_effective_weight(self, adapter_name: Optional[str] = None) -> torch.Tensor:
        rotation = self.get_input_rotation(adapter_name=adapter_name)
        base_weight = self.get_base_layer().weight.detach().to(rotation.dtype)
        return torch.matmul(base_weight, rotation.transpose(0, 1))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters or not self.active_adapters:
            return self.base_layer(x, *args, **kwargs)

        adapter_name = self.active_adapters[0]
        previous_dtype = x.dtype
        dropout = self.cara_dropout[adapter_name]
        x_dropped = dropout(x)

        u_matrix = torch.cat([self.cara_A[adapter_name], -self.cara_B[adapter_name]], dim=1).to(device=x.device)
        v_matrix = torch.cat([self.cara_B[adapter_name], self.cara_A[adapter_name]], dim=1).to(device=x.device)
        rank = self.r[adapter_name]

        identity_2r = torch.eye(2 * rank, device=x.device, dtype=torch.float32)
        inner_matrix = identity_2r + torch.matmul(v_matrix.transpose(0, 1).float(), u_matrix.float())
        z_matrix = torch.inverse(inner_matrix).to(x.dtype)
        u_matrix = u_matrix.to(x.dtype)
        v_matrix = v_matrix.to(x.dtype)

        rotated = x_dropped - 2 * torch.matmul(torch.matmul(torch.matmul(x_dropped, u_matrix), z_matrix), v_matrix.transpose(0, 1))


        interval = self.noise_step_interval[adapter_name]
        noise_alpha = self.noise_alpha[adapter_name]
        if self.training:
            self.step_counter += 1
            if interval > 0 and noise_alpha > 0.0 and self.step_counter % interval == 0:
                random_matrix = torch.randn(self.in_features, self.in_features, device=x.device, dtype=x.dtype)
                skew_matrix = (random_matrix - random_matrix.transpose(0, 1)) / math.sqrt(2 * self.in_features)
                rotated = rotated + noise_alpha * torch.matmul(rotated, skew_matrix)

        result = self.base_layer(x, *args, **kwargs)
        delta_x = rotated - x_dropped
        delta_out = F.linear(delta_x, self.get_base_layer().weight.to(x.dtype), bias=None)
        return (result + delta_out).to(previous_dtype)


def _matches_target_module(module_name: str, target_modules: Iterable[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def _resolve_submodule(root_module: nn.Module, module_path: str) -> nn.Module:
    module = root_module
    if not module_path:
        return module
    for part in module_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _get_submodules(root_module: nn.Module, module_name: str):
    parent_name, _, target_name = module_name.rpartition(".")
    parent = _resolve_submodule(root_module, parent_name)
    target = parent[int(target_name)] if target_name.isdigit() else getattr(parent, target_name)
    return parent, target, target_name


def _replace_submodule(parent: nn.Module, target_name: str, new_module: nn.Module):
    if target_name.isdigit():
        parent[int(target_name)] = new_module
    else:
        setattr(parent, target_name, new_module)


def set_cara_adapter(model: nn.Module, adapter_name: str = "default"):
    for module in model.modules():
        if isinstance(module, CaraLinear):
            module.set_adapter(adapter_name)


def inject_cara_adapter(model: nn.Module, config: CaraConfig, adapter_name: str = "default"):
    module_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches_target_module(name, config.target_modules)
    ]
    if not module_names:
        raise ValueError(f"No target modules matched {config.target_modules}")

    for module_name in module_names:
        parent, target, target_name = _get_submodules(model, module_name)
        if isinstance(target, CaraLinear):
            target.update_layer(
                adapter_name,
                r=config.r,
                cara_dropout=config.cara_dropout,
                noise_alpha=config.noise_alpha,
                noise_step_interval=config.noise_step_interval,
            )
            target.set_adapter(adapter_name)
            continue

        new_module = CaraLinear(
            target,
            adapter_name=adapter_name,
            r=config.r,
            cara_dropout=config.cara_dropout,
            noise_alpha=config.noise_alpha,
            noise_step_interval=config.noise_step_interval,
        )
        _replace_submodule(parent, target_name, new_module)

    if not hasattr(model, "cara_config"):
        model.cara_config = {}
    model.cara_config[adapter_name] = config
    model.active_cara_adapter = adapter_name
    set_cara_adapter(model, adapter_name=adapter_name)
    return model


def get_cara_model_state_dict(model: nn.Module, adapter_name: str = "default"):
    state_dict = model.state_dict()
    filtered_state = {}
    for key, value in state_dict.items():
        if f".cara_A.{adapter_name}" in key:
            filtered_state[key.replace(f".cara_A.{adapter_name}", ".cara_A")] = value
        elif f".cara_B.{adapter_name}" in key:
            filtered_state[key.replace(f".cara_B.{adapter_name}", ".cara_B")] = value
    return filtered_state


def set_cara_model_state_dict(model: nn.Module, cara_state_dict, adapter_name: str = "default"):
    remapped_state = {}
    for key, value in cara_state_dict.items():
        if ".cara_A" in key and f".cara_A.{adapter_name}" not in key:
            remapped_state[key.replace(".cara_A", f".cara_A.{adapter_name}")] = value
        elif ".cara_B" in key and f".cara_B.{adapter_name}" not in key:
            remapped_state[key.replace(".cara_B", f".cara_B.{adapter_name}")] = value
        else:
            remapped_state[key] = value
    return model.load_state_dict(remapped_state, strict=False)


def save_cara_adapter(model: nn.Module, save_directory: str, config: Optional[CaraConfig] = None, adapter_name: str = "default"):
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    config = config or getattr(model, "cara_config", {}).get(adapter_name)
    if config is None:
        raise ValueError("A CaraConfig is required to save a CARA adapter")
    config.save_pretrained(str(save_path))
    torch.save(get_cara_model_state_dict(model, adapter_name=adapter_name), save_path / WEIGHTS_NAME)


def load_cara_adapter(model: nn.Module, load_directory: str, adapter_name: str = "default", map_location: str = "cpu"):
    config = CaraConfig.from_pretrained(load_directory)
    if not any(isinstance(module, CaraLinear) for module in model.modules()):
        inject_cara_adapter(model, config, adapter_name=adapter_name)
    state_dict = torch.load(Path(load_directory) / WEIGHTS_NAME, map_location=map_location)
    load_result = set_cara_model_state_dict(model, state_dict, adapter_name=adapter_name)
    set_cara_adapter(model, adapter_name=adapter_name)
    return config, load_result


class CaraAdapterLayers(nn.Module):
    def __init__(self, model: nn.Module, adapter_name: str = "default"):
        super().__init__()
        object.__setattr__(self, "_cara_model_ref", model)
        self.adapter_name = adapter_name
        parameter_map = OrderedDict()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter_map[name.replace(".", "__")] = parameter
        if not parameter_map:
            raise ValueError("No CARA parameters were marked trainable")
        self.trainable_params = nn.ParameterDict(parameter_map)

    @property
    def model(self) -> nn.Module:
        return object.__getattribute__(self, "_cara_model_ref")

    def forward(self, *args, **kwargs):
        raise RuntimeError("CaraAdapterLayers is a checkpoint and DDP helper, not a forward model")

    def state_dict(self, *args, **kwargs):
        return get_cara_model_state_dict(self.model, adapter_name=self.adapter_name)

    def load_state_dict(self, state_dict, strict: bool = True):
        return set_cara_model_state_dict(self.model, state_dict, adapter_name=self.adapter_name)


def count_cara_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    return trainable_parameters, all_parameters