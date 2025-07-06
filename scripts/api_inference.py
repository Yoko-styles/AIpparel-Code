import os
from dataclasses import dataclass, field
import logging
from time import time

from pydantic import BaseModel 
log = logging.getLogger(__name__)
import hydra
from typing import Literal, Tuple, Any, Dict
import torch
import transformers
from typing import Optional
from torch.utils.data import DataLoader
from functools import partial
import torch.distributed as dist
from PIL import Image

from models.aipparel_model import AIpparelForCausalLM, AIpparelConfig
from models.llava import conversation as conversation_lib
from data.data_wrappers.collate_fns import collate_fn
from data.datasets.inference_dataset import InferenceDataset
from data.datasets.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN)
from trainers.utils import dict_to_cuda, dict_to_cpu, dict_to_dtype
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app = FastAPI(title="AIpparel Inference API", version="1.0.0")
# Add CORS middleware to accept all requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
inference_components = None
class InferenceRequest(BaseModel):
    user_input: str

class InferenceResponse(BaseModel):
    status: str
    patterns: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
@dataclass
class MainConfig:
    version: str
    model_max_length: int
    model: AIpparelConfig = field(default_factory=AIpparelConfig)
    precision: Literal["bf16", "fp16"] = "bf16"
    conv_type: Literal["default", "v0", "v1", "vicuna_v1", "llama_2", "plain", "v0_plain", "llava_v0", "v0_mmtag", "llava_v1", "v1_mmtag", "llava_llama_2", "mpt"] = "llava_v1"
    pre_trained: Optional[str] = None
    inference_json: str = "assets/data_configs/inference_example.json"
    vision_tower: str = "openai/clip-vit-large-patch14"
    panel_classification: str = "assets/data_configs/panel_classes_garmentcodedata.json"
    garment_tokenizer: str = "gcd_garment_tokenizer"


def load_model_and_tokenizer(cfg: MainConfig) -> Tuple[Any, Any, Any, torch.dtype]:
    """
    Load and initialize the model, tokenizer, and dataset once.
    Returns the loaded components for reuse.
    
    Returns:
        Tuple containing (model, tokenizer, dataset, torch_dtype)
    """
    log.info(f"Loading model from: {cfg.version}")
    
    # Create tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.version,
        cache_dir=None,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Create dataset
    dataset = InferenceDataset(
        inference_json=cfg.inference_json,
        vision_tower=cfg.vision_tower,
        image_size=224,
        garment_tokenizer=hydra.utils.instantiate(cfg.garment_tokenizer),
        panel_classification=cfg.panel_classification
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    all_new_tokens = dataset.get_all_token_names()
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    log.info(f"Added {num_added_tokens} tokens to the tokenizer.")
    
    token_name2_idx_dict = {}
    for token in all_new_tokens:
        token_idx = tokenizer(token, add_special_tokens=False).input_ids[0]
        token_name2_idx_dict[token] = token_idx
        
    log.info(f"Token name to index dictionary size: {len(token_name2_idx_dict)}")
    dataset.set_token_indices(token_name2_idx_dict)

    if cfg.model.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    # Set torch dtype
    torch_dtype = torch.float32
    if cfg.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.precision == "fp16":
        torch_dtype = torch.half
        
    # Create model
    model = AIpparelForCausalLM.from_pretrained(
        cfg.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **cfg.model, vision_tower=cfg.vision_tower, 
        panel_edge_indices=dataset.panel_edge_type_indices, gt_stats=dataset.gt_stats
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device="cuda")
    model.resize_token_embeddings(len(tokenizer))

    for p in model.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[cfg.conv_type]
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params}")
    
    # Load pre-trained weights
    assert cfg.pre_trained is not None
    state_dict = torch.load(cfg.pre_trained, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda")
    model.eval()
    
    log.info("Model loaded successfully!")
    return model, tokenizer, dataset, torch_dtype


def run_inference(model, tokenizer, dataset, torch_dtype, cfg: MainConfig, custom_prompt: str) -> Tuple[str, Any, Dict]:
    """
    Run inference on a custom prompt using the loaded model.
    This function can be called multiple times with different prompts.
    
    Args:
        model: Loaded AIpparel model
        tokenizer: Configured tokenizer
        dataset: Dataset instance
        torch_dtype: Torch data type for precision
        cfg: Configuration object
        custom_prompt: Custom description to process
        
    Returns:
        Tuple containing (output_text, patterns, input_dict)
    """
    log.info(f"Running inference on: {custom_prompt}")
    
    # Override the dataset with custom prompt
    dataset.datapoints_names = [("description", {"description": custom_prompt})]
    
    # Create data loader
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=cfg.conv_type,
            use_mm_start_end=cfg.model.use_mm_start_end,
            local_rank=0,
            generation_only=True,
        )
    )
    
    # Process the single batch
    for input_dict in val_loader:
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        input_dict = dict_to_dtype(input_dict,
            torch_dtype,
            [
                "images_clip",
                "param_targets",
                "param_target_endpoints",
                "param_target_transformations",
                "questions_pattern_endpoints",
                "questions_pattern_transformations"
            ]
        )

        # Run model evaluation
        output_dict = model.evaluate(
            input_dict["images_clip"],
            input_dict["question_ids"],
            input_dict["question_attention_masks"],
            endpoints=input_dict["questions_pattern_endpoints"],
            endpoints_mask=input_dict["questions_pattern_endpoints_mask"],
            transformations=input_dict["questions_pattern_transformations"],
            transformations_mask=input_dict["questions_pattern_transformations_mask"],
            max_new_tokens=2100
        )
        
        output_dict = dict_to_cpu(output_dict)
        output_dict = dict_to_dtype(output_dict, torch.float32)
        output_dict["input_mask"] = torch.arange(output_dict["output_ids"].shape[1]).reshape(1, -1) >= input_dict["question_ids"].shape[1]
        output_text, patterns, _ = dataset.decode(output_dict, tokenizer)
        
        log.info("Inference completed successfully!")
        return output_text, patterns, input_dict

@app.post("/inference", response_model=InferenceResponse)
async def api_run_inference(request: InferenceRequest):
    """Run inference on user input"""
    global inference_components
    
    if inference_components is None:
        return InferenceResponse(
            status="error",
            error="Model components not initialized. Please initialize the model first."
        )
    
    try:
        tokenizer, model, dataset, torch_dtype, cfg = inference_components
        prompt = request.user_input.strip()
        # output_text, patterns, input_dict = run_inference(model, tokenizer, dataset, torch_dtype, cfg, prompt)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        save_path = f'{output_dir}/inference'
        os.makedirs(save_path, exist_ok=True)
        patterns = None
        try:
            # Run inference - this is fast since model is already loaded
            output_text, patterns, input_dict = run_inference(model, tokenizer, dataset, torch_dtype, cfg, prompt)
         
            # Save results
            timestamp = int(time())
            data_name = f"sample_{timestamp}"
            os.makedirs(os.path.join(save_path, data_name), exist_ok=True)
            patterns.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_pred')
            
            if "gt_patterns" in input_dict:
                for gt_pattern in input_dict["gt_patterns"][0]:
                    gt_pattern.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_gt')
            
            # Save output text
            with open(os.path.join(save_path, data_name, "output.txt"), "w") as f:
                question = input_dict["questions_list"][0]
                f.write(f"Question: {question}\n")
                f.write(f"Output Text: {output_text}\n")
                f.write(f"Custom Prompt: {prompt}\n")

            # Save input image if exists
            if os.path.isfile(input_dict["image_paths"][0]):
                cond_img = Image.open(input_dict["image_paths"][0])
                cond_img.save(os.path.join(save_path, data_name, 'input.png'))
            log.info(f"Results for prompt saved to: {os.path.join(save_path, data_name)}")
            
        except Exception as e:
            log.error(f"Error processing prompt '{prompt}': {e}")

        return InferenceResponse(
            status="success",
            patterns= patterns.spec
        )

        
    
    except Exception as e:
        log.error(f"Error during inference: {str(e)}")
        return InferenceResponse(
            status="error",
            error=f"Inference failed: {str(e)}"
        )

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: MainConfig):
    global inference_components

    ddp_rank = int(os.environ.get('RANK', 0))
    master_process = (ddp_rank == 0)
    
    # Load model once
    model, tokenizer, dataset, torch_dtype = load_model_and_tokenizer(cfg)
    inference_components = (tokenizer, model, dataset, torch_dtype, cfg)
    
    if master_process:
        log.info("Starting FastAPI server on port 8000 (master process)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        log.info(f"Worker process (rank {ddp_rank}) initialized, waiting...")
        # Keep the worker process alive
        import time
        while True:
            time.sleep(60) 
             
    # log.info(f"Working directory : {os.getcwd()}")
    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # log.info(f"Output directory : {output_dir}")

    # # Example: Run multiple inferences with different prompts
    # custom_prompts = [
    #     "A dress with a square neckline and a pleated skirt",
    #     "A flowing evening gown with off-shoulder sleeves",
    #     "A casual summer dress with floral patterns",
    #     "A vintage-inspired blouse with puffed sleeves"
    # ]
    
    # save_path = f'{output_dir}/inference'
    # os.makedirs(save_path, exist_ok=True)
    
    # for i, prompt in enumerate(custom_prompts):
    #     try:
    #         # Run inference - this is fast since model is already loaded
    #         output_text, patterns, input_dict = run_inference(model, tokenizer, dataset, torch_dtype, cfg, prompt)
            
    #         # Save results
    #         data_name = f"sample_{i}"
    #         os.makedirs(os.path.join(save_path, data_name), exist_ok=True)
    #         patterns.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_pred')
            
    #         if "gt_patterns" in input_dict:
    #             for gt_pattern in input_dict["gt_patterns"][0]:
    #                 gt_pattern.serialize(os.path.join(save_path, data_name), spec_only=False, with_3d=False, with_text=False, view_ids=False, to_subfolder=False, tag=f'_gt')
            
    #         # Save output text
    #         with open(os.path.join(save_path, data_name, "output.txt"), "w") as f:
    #             question = input_dict["questions_list"][0]
    #             f.write(f"Question: {question}\n")
    #             f.write(f"Output Text: {output_text}\n")
    #             f.write(f"Custom Prompt: {prompt}\n")
            
    #         # Save input image if exists
    #         if os.path.isfile(input_dict["image_paths"][0]):
    #             cond_img = Image.open(input_dict["image_paths"][0])
    #             cond_img.save(os.path.join(save_path, data_name, 'input.png'))
                
    #         log.info(f"Results for prompt {i+1} saved to: {os.path.join(save_path, data_name)}")
            
    #     except Exception as e:
    #         log.error(f"Error processing prompt '{prompt}': {e}")


# Example usage for API calls
def initialize_model(cfg: MainConfig):
    """Initialize model for API usage"""
    return load_model_and_tokenizer(cfg)

def predict(model, tokenizer, dataset, torch_dtype, cfg: MainConfig, prompt: str):
    """Single prediction function for API usage"""
    return run_inference(model, tokenizer, dataset, torch_dtype, cfg, prompt)


if __name__ == "__main__":
    main()
