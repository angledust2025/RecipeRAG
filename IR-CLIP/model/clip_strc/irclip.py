import torch
import torch.nn as nn
import math
from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union
from .modeling_clip import CLIPModel, CLIPTextTransformer, CLIPVisionTransformer, CLIPOutput, CLIPAttention, CLIPMLP

import torch.distributed as dist

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch import nn

import math


class IRCLIPModel(CLIPModel):
    config_class = CLIPConfig
    main_input_name = "text_long"

    def __init__(self, config):
        super(CLIPModel, self).__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        text_config.eos_token_id = 49407
        text_config.pad_token_id = 49407
        text_config.bos_token_id = 49406

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)

        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)


        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.title_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.ingredients_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.instructions_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_finegraind = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))


        self.embed_dim = text_config.hidden_size
        self.world_size = 0

        # Initialize weights and apply final processing
        self.post_init()


    # ... (resize_postion_embeding 和 copy_weight 函数保持不变，省略以节省篇幅) ...
    def resize_postion_embeding(self, newsize=248):
        # ... (保持原样) ...
        old_position_embedding = self.text_model.embeddings.position_embedding
        old_position_embedding_res = self.text_model.embeddings.position_embedding_res
        old_position_embedding_ori = self.text_model.embeddings.position_embedding_ori
        
        positional_embedding_pre = self.text_model.embeddings.position_embedding.weight.data
    
        length, dim = positional_embedding_pre.shape
        keep_len = 20
        posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=positional_embedding_pre.dtype)
        for i in range(keep_len):
            posisitonal_embedding_new[i] = positional_embedding_pre[i]
        for i in range(length-1-keep_len):
            posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

        posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
                
        positional_embedding_res = posisitonal_embedding_new.clone()

        self.text_model.embeddings.position_embedding_ori.weight.data = posisitonal_embedding_new
        self.text_model.embeddings.position_embedding_ori.num_embeddings = posisitonal_embedding_new.shape[0]
        
        self.text_model.embeddings.position_embedding_res.weight.data = positional_embedding_res
        self.text_model.embeddings.position_embedding_res.num_embeddings = positional_embedding_res.shape[0]

        old_position_embedding_ori_requires_grad = old_position_embedding_ori.weight.requires_grad
        self.text_model.embeddings.position_embedding_ori.requires_grad_(old_position_embedding_ori_requires_grad)

        old_position_embedding_res_requires_grad = old_position_embedding_res.weight.requires_grad
        self.text_model.embeddings.position_embedding_res.requires_grad_(old_position_embedding_res_requires_grad)

    def copy_weight(self,):
        with torch.no_grad():
            self.title_projection.weight.data.copy_(self.text_projection.weight.data)  
            self.ingredients_projection.weight.data.copy_(self.text_projection.weight.data)  
            self.instructions_projection.weight.data.copy_(self.text_projection.weight.data)

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features
    

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        walk_short_pos: Optional[bool] = True,
        use_bbox: Optional[bool] = False,
        mode: Optional[str] = 'instructions',
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pos_flag = walk_short_pos or use_bbox

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            walk_short_pos=pos_flag,
            mode=mode,
        )
        pooled_output = text_outputs[1]

        if walk_short_pos:
            if mode == 'title': 
                text_features = self.title_projection(pooled_output)
            elif mode == 'ingredients':
                text_features = self.ingredients_projection(pooled_output)
            elif mode == 'instructions':
                text_features = self.instructions_projection(pooled_output)
            else:
                raise ValueError(f"Unsupported text mode: {mode!r}")
        else:
            text_features = self.instructions_projection(pooled_output)    

        return text_features


    def forward(
        self,
        instructions_texts: Optional[torch.LongTensor] = None,
        ingredients_texts: Optional[torch.LongTensor] = None,
        title_texts: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        seg_image: Optional[torch.FloatTensor] = None,
        hard_inst_texts: Optional[torch.LongTensor] = None,
        hard_ingr_texts: Optional[torch.LongTensor] = None,
        hard_title_texts: Optional[torch.LongTensor] = None,
        hard_nums: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,      # 新增
        output_attentions: Optional[bool] = None, # 确保有这行
        output_hidden_states: Optional[bool] = None, # 确保有这行
        return_dict: Optional[bool] = None,      # 确保有这行
        add_seg_loss: bool = False,
        use_hard_neg: bool = False,
        **kwargs # 加上这个可以防止 transformers 传入多余参数时报错
    ) -> Union[Tuple, CLIPOutput]:


        output_attentions = True 
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
         
        # 1. 计算原图 (Image) 的特征
        vision_outputs = self.vision_model(
                pixel_values=image,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=return_dict,
        )
        
        # 2. 文本特征计算 (保持不变)
        instructions_text_outputs = self.text_model(
                input_ids=instructions_texts,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_short_pos=True,
                mode="instructions",
            )

        ingredients_text_outputs = self.text_model(
                input_ids=ingredients_texts,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_short_pos=True,
                mode="ingredients",
            )

        title_text_outputs = self.text_model(
                input_ids=title_texts,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                walk_short_pos=True,
                mode="title",
            )

        instructions_text_embeds = instructions_text_outputs[1]
        instructions_text_embeds = self.instructions_projection(instructions_text_embeds)
        instructions_text_embeds = instructions_text_embeds / instructions_text_embeds.norm(p=2, dim=-1, keepdim=True)

        ingredients_text_embeds = ingredients_text_outputs[1]
        ingredients_text_embeds = self.ingredients_projection(ingredients_text_embeds)
        ingredients_text_embeds = ingredients_text_embeds / ingredients_text_embeds.norm(p=2, dim=-1, keepdim=True)

        title_text_embeds = title_text_outputs[1]
        title_text_embeds = self.title_projection(title_text_embeds)
        title_text_embeds = title_text_embeds / title_text_embeds.norm(p=2, dim=-1, keepdim=True)


        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # ==========================================
        # 全局对齐
        # ==========================================
        global_text_embeds = (instructions_text_embeds + ingredients_text_embeds + title_text_embeds) / 3.0
        global_text_embeds = global_text_embeds / global_text_embeds.norm(p=2, dim=-1, keepdim=True)

        loss_ins, loss_ing, loss_tit = self.clip_loss(image_embeds, instructions_text_embeds, ingredients_text_embeds, title_text_embeds, rank, image)
        loss_global = self.clip_global_loss(image_embeds, global_text_embeds, rank)

        # 基础 Loss
        loss = loss_global + 0.3*(loss_tit + loss_ing + loss_ins)

        # ==========================================
        # 分割损失
        # ==========================================
        loss_seg_align = torch.tensor(0.0, device=image.device)
        
        if add_seg_loss and seg_image is not None:
            # 1. 计算去背景图 (seg_image) 的特征
            seg_vision_outputs = self.vision_model(
                pixel_values=seg_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            seg_image_embeds = seg_vision_outputs[1]
            seg_image_embeds = self.visual_projection(seg_image_embeds)
            seg_image_embeds = seg_image_embeds / seg_image_embeds.norm(p=2, dim=-1, keepdim=True)
            

            loss_seg_align = self.clip_global_loss(seg_image_embeds, global_text_embeds, rank)
            
            loss = loss + loss_seg_align



        if rank == 0:
            print(
                f"[Step Loss Info] "
                f"Total: {loss.item():.4f} | "
                f"Global: {loss_global.item():.4f} | "
                f"SegAlign: {loss_seg_align.item():.4f} | "  # 监控这个 Loss 是否正常下降
                f"Tit: {loss_tit.item():.4f} | "
                f"Ing: {loss_ing.item():.4f} | "
                f"Ins: {loss_ins.item():.4f} "
            )

        return CLIPOutput(loss,)


    def clip_loss(self,image_features_long, text_features_instructions, text_features_ingredients, text_features_title, rank, image):
        image_feat_all_long = torch.cat(nn_dist.all_gather(image_features_long), dim=0) if dist.is_available() and dist.is_initialized() else image_features_long
        text_feat_all_instructions = torch.cat(nn_dist.all_gather(text_features_instructions), dim=0) if dist.is_available() and dist.is_initialized() else text_features_instructions
        text_feat_all_ingredients = torch.cat(nn_dist.all_gather(text_features_ingredients), dim=0) if dist.is_available() and dist.is_initialized() else text_features_ingredients
        text_feat_all_title = torch.cat(nn_dist.all_gather(text_features_title), dim=0) if dist.is_available() and dist.is_initialized() else text_features_title
        
        sim_i2ins = torch.matmul(image_features_long, text_feat_all_instructions.T)
        sim_ins2i = torch.matmul(image_feat_all_long, text_features_instructions.T)
        sim_ins2i = sim_ins2i.T

        sim_i2ing = torch.matmul(image_features_long, text_feat_all_ingredients.T)
        sim_ing2i = torch.matmul(image_feat_all_long, text_features_ingredients.T)
        sim_ing2i = sim_ing2i.T

        sim_i2tit = torch.matmul(image_features_long, text_feat_all_title.T)
        sim_tit2i = torch.matmul(image_feat_all_long, text_features_title.T)
        sim_tit2i = sim_tit2i.T
        
        sim_i2ins = self.logit_scale_finegraind.exp() * sim_i2ins
        sim_ins2i = self.logit_scale_finegraind.exp() * sim_ins2i

        sim_i2ing = self.logit_scale_finegraind.exp() * sim_i2ing
        sim_ing2i = self.logit_scale_finegraind.exp() * sim_ing2i

        sim_i2tit = self.logit_scale_finegraind.exp() * sim_i2tit
        sim_tit2i = self.logit_scale_finegraind.exp() * sim_tit2i
        
        bs = image_features_long.size(0)
        targets = torch.linspace(rank * bs,rank * bs + bs - 1, bs, dtype=torch.long).to(image.device)

        loss_ins = (
                    F.cross_entropy(sim_i2ins, targets, label_smoothing=0.0)
                    + F.cross_entropy(sim_ins2i, targets, label_smoothing=0.0)
                ) / 2
        
        loss_ing = (
                F.cross_entropy(sim_i2ing, targets, label_smoothing=0.0)
                + F.cross_entropy(sim_ing2i, targets, label_smoothing=0.0)
            ) / 2

        loss_tit = (
                F.cross_entropy(sim_i2tit, targets, label_smoothing=0.0)
                + F.cross_entropy(sim_tit2i, targets, label_smoothing=0.0)
            ) / 2

        return loss_ins, loss_ing, loss_tit
    

    def clip_global_loss(self, image_features, text_features, rank):

        image_feat_all = torch.cat(nn_dist.all_gather(image_features), dim=0) if dist.is_available() and dist.is_initialized() else image_features
        text_feat_all = torch.cat(nn_dist.all_gather(text_features), dim=0) if dist.is_available() and dist.is_initialized() else text_features
        

        sim_i2t = torch.matmul(image_features, text_feat_all.T) * self.logit_scale.exp()
        sim_t2i = torch.matmul(image_feat_all, text_features.T).T * self.logit_scale.exp()
        
        bs = image_features.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(image_features.device)
        
        loss = (F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)) / 2
        return loss



