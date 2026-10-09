# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: Optional[str] = None
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `gdpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "low_var_kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    gdpo_reward_keys: Tuple[str, ...] = ("title", "ingredients", "instructions_rouge", "instructions_bleu")
    """reward metric keys used as GDPO objectives"""
    gdpo_reward_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    """objective weights used in GDPO weighted sum"""
    gdpo_group_eps: float = 1e-6
    """epsilon for GDPO objective-wise group normalization"""
    gdpo_batch_norm: bool = False
    """whether to apply GDPO batch-level normalization after objective sum"""
    gdpo_batch_eps: float = 1e-6
    """epsilon for GDPO batch-level normalization"""
    gdpo_pareto_enabled: bool = False
    """whether to optimize GDPO objectives with Pareto-style gradient coordination instead of scalar summation"""
    gdpo_pareto_solver: str = "mgda"
    """Pareto solver for GDPO objectives, currently support `mgda`"""
    gdpo_pareto_pref: Tuple[float, ...] = ()
    """optional preference vector for selecting one Pareto trade-off point; empty means uniform"""
    gdpo_pareto_pref_reg: float = 1e-2
    """regularization strength that keeps Pareto weights close to the preference vector"""
    gdpo_pareto_max_iter: int = 25
    """maximum projected-gradient iterations for the Pareto simplex solver"""
    gdpo_pareto_tol: float = 1e-6
    """tolerance for the Pareto simplex solver"""
    gdpo_ema_decay: float = 0.9
    """EMA decay factor for loss-adaptive objective weighting in GDPO Pareto mode"""
    gdpo_ema_temperature: float = 0.05
    """softmax temperature for EMA-based alpha: smaller = more aggressive reweighting"""
    gdpo_alpha_mode: str = "loss_ema"
    """objective weighting mode in GDPO Pareto mode: `loss_ema` or `reward_improvement`"""
    gdpo_reward_targets: Tuple[float, ...] = ()
    """target reward values used by reward-improvement GDPO objective weighting"""
    gdpo_reward_fast_ema_decay: float = 0.8
    """fast EMA decay for reward-improvement GDPO objective weighting"""
    gdpo_reward_slow_ema_decay: float = 0.98
    """slow EMA decay for reward-improvement GDPO objective weighting"""
    gdpo_alpha_temperature: float = 0.5
    """softmax temperature for reward-improvement GDPO objective weights; larger = softer distribution"""
    gdpo_alpha_headroom_lambda: float = 0.8
    """blend for reward-improvement GDPO priority: 1.0 only uses improvement, 0.0 only uses target headroom"""
    gdpo_alpha_min: float = 0.0
    """minimum per-objective GDPO alpha before renormalization; 0 disables the lower bound"""
    gdpo_alpha_max: float = 1.0
    """maximum per-objective GDPO alpha before renormalization; 1 disables the upper bound"""
    gdpo_alpha_eps: float = 1e-6
    """epsilon for reward-improvement GDPO objective weighting"""
    gdpo_curriculum_target_headroom: float = 0.0
    """curriculum learning headroom for GDPO reward targets; when > 0, dynamic_target = max(fixed_target,
    slow_ema + headroom), so the target keeps rising as the model improves; 0 disables curriculum"""
    gdpo_loss_norm_ema_decay: float = 0.9
    """EMA decay for per-objective pg_loss scale normalization; normalises each objective's pg_loss by its
    running absolute-value EMA so all objectives contribute equal magnitude to the gradient regardless of
    their natural scale differences; set to 0.0 to disable"""
    entropy_coef: float = 0.0
    """entropy regularization coefficient added as -entropy_coef * H(π) to the total loss to prevent
    entropy collapse; 0 disables; typical values 0.001-0.005"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.gdpo_reward_keys = self.algorithm.gdpo_reward_keys
        self.worker.actor.gdpo_pareto_enabled = self.algorithm.gdpo_pareto_enabled
        self.worker.actor.gdpo_pareto_solver = self.algorithm.gdpo_pareto_solver
        self.worker.actor.gdpo_pareto_pref = self.algorithm.gdpo_pareto_pref
        self.worker.actor.gdpo_pareto_pref_reg = self.algorithm.gdpo_pareto_pref_reg
        self.worker.actor.gdpo_pareto_max_iter = self.algorithm.gdpo_pareto_max_iter
        self.worker.actor.gdpo_pareto_tol = self.algorithm.gdpo_pareto_tol
        self.worker.actor.gdpo_ema_decay = self.algorithm.gdpo_ema_decay
        self.worker.actor.gdpo_ema_temperature = self.algorithm.gdpo_ema_temperature
        self.worker.actor.gdpo_alpha_mode = self.algorithm.gdpo_alpha_mode
        self.worker.actor.entropy_coef = self.algorithm.entropy_coef
        self.worker.actor.gdpo_loss_norm_ema_decay = self.algorithm.gdpo_loss_norm_ema_decay

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
