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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss, solve_pareto_weights
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
        # EMA state for loss-adaptive objective weighting (GDPO Pareto mode)
        self._pg_loss_ema: Optional[torch.Tensor] = None
        # EMA of per-objective |pg_loss| used for scale normalisation
        self._pg_loss_scale_ema: Optional[torch.Tensor] = None

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    def _get_pareto_preference(self, num_objectives: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if len(self.config.gdpo_pareto_pref) == 0:
            return torch.full((num_objectives,), 1.0 / num_objectives, device=device, dtype=dtype)

        if len(self.config.gdpo_pareto_pref) != num_objectives:
            raise ValueError(
                f"GDPO Pareto preference size mismatch: expected {num_objectives}, got {len(self.config.gdpo_pareto_pref)}."
            )
        preference = torch.as_tensor(self.config.gdpo_pareto_pref, device=device, dtype=dtype)
        preference = torch.clamp(preference, min=0.0)
        preference_sum = preference.sum()
        if preference_sum <= 0:
            return torch.full((num_objectives,), 1.0 / num_objectives, device=device, dtype=dtype)
        return preference / preference_sum

    def _compute_pareto_alpha(self, objective_losses: list[torch.Tensor], log_probs: torch.Tensor) -> torch.Tensor:
        num_objectives = len(objective_losses)
        if num_objectives == 1:
            return torch.ones(1, device=log_probs.device, dtype=log_probs.dtype)

        grad_vectors = []
        for objective_loss in objective_losses:
            grad_log_probs = torch.autograd.grad(
                objective_loss,
                log_probs,
                retain_graph=True,
                allow_unused=False,
            )[0]
            grad_vectors.append(grad_log_probs.detach().reshape(-1))

        grad_matrix = torch.stack(grad_vectors, dim=0)
        gram_matrix = grad_matrix @ grad_matrix.transpose(0, 1)
        if dist.is_initialized():
            dist.all_reduce(gram_matrix, op=dist.ReduceOp.SUM)

        if self.config.gdpo_pareto_solver != "mgda":
            raise NotImplementedError(f"Unknown GDPO Pareto solver: {self.config.gdpo_pareto_solver}.")

        preference = self._get_pareto_preference(num_objectives, device=gram_matrix.device, dtype=gram_matrix.dtype)
        return solve_pareto_weights(
            gram_matrix=gram_matrix,
            preference=preference,
            regularization=self.config.gdpo_pareto_pref_reg,
            max_iter=self.config.gdpo_pareto_max_iter,
            tol=self.config.gdpo_pareto_tol,
        )

    def _compute_ema_alpha(self, objective_losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute objective weights using EMA-smoothed pg_loss values.

        Objectives with consistently negative pg_loss (being suppressed) receive
        higher weight; objectives with large positive pg_loss (over-optimized) receive
        lower weight. EMA smoothing prevents single-step noise from causing instability.

        Args:
            objective_losses: per-objective pg_loss tensors (unscaled), one per objective.

        Returns:
            alpha: weight tensor of shape (num_objectives,), sums to 1.
        """
        num_objectives = len(objective_losses)
        if num_objectives == 1:
            return torch.ones(1, device=objective_losses[0].device, dtype=objective_losses[0].dtype)

        device = objective_losses[0].device

        # Collect current pg_loss values (detached scalars, no grad needed)
        current_pg = torch.tensor(
            [loss.detach().item() for loss in objective_losses],
            dtype=torch.float32,
            device=device,
        )

        # Synchronize across processes so all ranks share the same EMA
        if dist.is_initialized():
            dist.all_reduce(current_pg, op=dist.ReduceOp.SUM)
            current_pg = current_pg / dist.get_world_size()

        # Update EMA: ema = decay * ema_prev + (1 - decay) * current
        decay = self.config.gdpo_ema_decay
        if self._pg_loss_ema is None:
            self._pg_loss_ema = current_pg.clone()
        else:
            self._pg_loss_ema = self._pg_loss_ema.to(device=device)
            self._pg_loss_ema = decay * self._pg_loss_ema + (1.0 - decay) * current_pg

        # alpha = softmax(-ema / T): suppressed objectives (negative ema) get higher weight
        T = self.config.gdpo_ema_temperature
        alpha = torch.softmax(-self._pg_loss_ema / T, dim=0)
        return alpha.to(dtype=objective_losses[0].dtype)

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        if self.config.gdpo_pareto_enabled:
            select_keys.append("gdpo_objective_advantages")
            if self.config.gdpo_alpha_mode == "reward_improvement":
                select_keys.append("gdpo_objective_alpha")
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    objective_advantages = model_inputs.get("gdpo_objective_advantages")
                    objective_alpha = model_inputs.get("gdpo_objective_alpha")

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    micro_batch_scale = torch.sum(response_mask) * self.world_size / total_response_tokens

                    if self.config.gdpo_pareto_enabled and objective_advantages is not None:
                        num_objectives = objective_advantages.size(1)
                        if self.config.gdpo_reward_keys and len(self.config.gdpo_reward_keys) != num_objectives:
                            raise ValueError(
                                "GDPO reward keys size mismatch for Pareto-GDPO: "
                                f"expected {num_objectives}, got {len(self.config.gdpo_reward_keys)}."
                            )

                        objective_pg_losses = []
                        objective_pg_metrics = []
                        reward_keys = (
                            self.config.gdpo_reward_keys
                            if len(self.config.gdpo_reward_keys) == num_objectives
                            else tuple(f"objective_{i}" for i in range(num_objectives))
                        )
                        for objective_idx, reward_key in enumerate(reward_keys):
                            objective_pg_loss, objective_metrics = compute_policy_loss(
                                old_log_probs=old_log_probs,
                                log_probs=log_probs,
                                advantages=objective_advantages[:, objective_idx, :],
                                response_mask=response_mask,
                                clip_ratio_low=self.config.clip_ratio_low,
                                clip_ratio_high=self.config.clip_ratio_high,
                                clip_ratio_dual=self.config.clip_ratio_dual,
                                tau_positive=self.config.tau_positive,
                                tau_negative=self.config.tau_negative,
                                loss_type=self.config.loss_type,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            objective_pg_losses.append(objective_pg_loss)
                            objective_pg_metrics.append((reward_key, objective_metrics))

                        if self.config.gdpo_alpha_mode == "reward_improvement":
                            if objective_alpha is None:
                                raise ValueError("GDPO reward-improvement alpha mode requires `gdpo_objective_alpha`.")
                            if objective_alpha.ndim == 1:
                                pareto_alpha = objective_alpha.to(device=log_probs.device, dtype=log_probs.dtype)
                            elif objective_alpha.ndim == 2:
                                if objective_alpha.size(-1) != num_objectives:
                                    raise ValueError(
                                        "GDPO objective alpha size mismatch: "
                                        f"expected {num_objectives}, got {objective_alpha.size(-1)}."
                                    )
                                pareto_alpha = objective_alpha.to(device=log_probs.device, dtype=log_probs.dtype).mean(dim=0)
                            else:
                                raise ValueError(f"GDPO objective alpha expects 1D or 2D tensor, got {tuple(objective_alpha.shape)}.")

                            pareto_alpha = pareto_alpha / pareto_alpha.sum().clamp_min(1e-8)
                        elif self.config.gdpo_alpha_mode == "loss_ema":
                            pareto_alpha = self._compute_ema_alpha(objective_pg_losses)
                        else:
                            raise NotImplementedError(f"Unknown GDPO alpha mode: {self.config.gdpo_alpha_mode}.")

                        # --- pg_loss scale normalisation ---
                        # Each objective can have very different natural loss magnitudes (e.g. rouge
                        # pg_loss is ~5x larger than title pg_loss at init). Without normalisation,
                        # high-magnitude objectives dominate the gradient even when alpha is equal.
                        # We maintain a running EMA of |pg_loss_k| and divide each loss by its EMA
                        # before alpha-weighting, so every objective contributes equal gradient scale.
                        _loss_norm_decay = float(getattr(self.config, "gdpo_loss_norm_ema_decay", 0.0))
                        if _loss_norm_decay > 0.0:
                            _current_scales = torch.tensor(
                                [loss.detach().abs().item() for loss in objective_pg_losses],
                                dtype=torch.float32,
                                device=log_probs.device,
                            )
                            if dist.is_initialized():
                                dist.all_reduce(_current_scales, op=dist.ReduceOp.SUM)
                                _current_scales = _current_scales / dist.get_world_size()

                            if self._pg_loss_scale_ema is None:
                                self._pg_loss_scale_ema = _current_scales.clone()
                            else:
                                self._pg_loss_scale_ema = self._pg_loss_scale_ema.to(device=log_probs.device)
                                self._pg_loss_scale_ema = (
                                    _loss_norm_decay * self._pg_loss_scale_ema
                                    + (1.0 - _loss_norm_decay) * _current_scales
                                )

                            _scale_eps = 1e-8
                            _losses_to_combine = [
                                loss / (self._pg_loss_scale_ema[k] + _scale_eps)
                                for k, loss in enumerate(objective_pg_losses)
                            ]
                        else:
                            _losses_to_combine = objective_pg_losses

                        pg_loss = torch.sum(
                            torch.stack(_losses_to_combine)
                            * pareto_alpha.to(device=log_probs.device, dtype=log_probs.dtype)
                        )

                        for objective_idx, reward_key in enumerate(reward_keys):
                            ema_val = (
                                self._pg_loss_ema[objective_idx].item()
                                if self._pg_loss_ema is not None
                                else float("nan")
                            )
                            _scale_ema_val = (
                                self._pg_loss_scale_ema[objective_idx].item()
                                if self._pg_loss_scale_ema is not None
                                else float("nan")
                            )
                            append_to_dict(
                                metrics,
                                {
                                    f"actor/gdpo_{reward_key}/pg_loss": objective_pg_losses[objective_idx].detach().item(),
                                    f"actor/gdpo_{reward_key}/alpha": pareto_alpha[objective_idx].detach().item(),
                                    f"actor/gdpo_{reward_key}/ema_pg_loss": ema_val,
                                    f"actor/gdpo_{reward_key}/pg_loss_scale_ema": _scale_ema_val,
                                    f"actor/gdpo_{reward_key}/ppo_kl": objective_pg_metrics[objective_idx][1]["ppo_kl"],
                                    f"actor/gdpo_{reward_key}/entropy_loss": objective_pg_metrics[objective_idx][1]["entropy_loss"],
                                },
                            )
                            if "pg_clipfrac_higher" in objective_pg_metrics[objective_idx][1]:
                                append_to_dict(
                                    metrics,
                                    {
                                        f"actor/gdpo_{reward_key}/pg_clipfrac_higher": objective_pg_metrics[objective_idx][1]["pg_clipfrac_higher"],
                                        f"actor/gdpo_{reward_key}/pg_clipfrac_lower": objective_pg_metrics[objective_idx][1]["pg_clipfrac_lower"],
                                    },
                                )
                    else:
                        pg_loss, pg_metrics = compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            tau_positive=self.config.tau_positive,
                            tau_negative=self.config.tau_negative,
                            loss_type=self.config.loss_type,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                        batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                        append_to_dict(metrics, batch_metrics)

                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        append_to_dict(
                            metrics,
                            {
                                "actor/kl_loss": kl_loss.detach().item(),
                                "actor/kl_coef": self.config.kl_coef,
                            },
                        )
                    else:
                        loss = pg_loss

                    # --- entropy regularisation ---
                    # H(π) = E[-log π] ≥ 0. Adding -entropy_coef * H(π) to the loss
                    # maximises entropy, preventing the policy from collapsing to a
                    # near-deterministic output distribution (entropy collapse).
                    _entropy_coef = float(getattr(self.config, "entropy_coef", 0.0))
                    if _entropy_coef > 0.0:
                        # average_loss(-log_probs, ...) = E[-log π] = H(π), differentiable
                        _entropy = average_loss(-log_probs, response_mask, mode=self.config.loss_avg_mode)
                        loss = loss - _entropy_coef * _entropy
                        append_to_dict(
                            metrics,
                            {
                                "actor/entropy_bonus": _entropy.detach().item(),
                                "actor/entropy_coef": _entropy_coef,
                            },
                        )

                    loss = loss * micro_batch_scale
                    loss.backward()
                    append_to_dict(metrics, {"actor/pg_loss": pg_loss.detach().item()})

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
