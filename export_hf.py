import os
import json
import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import argparse

def main():
    parser = argparse.ArgumentParser(description="Export NanoRushGPT to Hugging Face format")
    parser.add_argument("--model_dir", type=str, default="models/nano-chat", help="Input directory containing custom model")
    parser.add_argument("--out_dir", type=str, default="models/hf-nano-chat", help="Output directory for Hugging Face model")
    args = parser.parse_args()

    model_dir = args.model_dir
    out_dir = args.out_dir
    
    if not os.path.exists(model_dir):
        print(f"Error: {model_dir} not found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("Loading custom config...")
    with open(f"{model_dir}/config.json") as f:
        custom_config = json.load(f)

    print("Creating Hugging Face GPT2Config...")
    hf_config = GPTConfig(
        vocab_size=custom_config["vocab_size"],
        n_positions=custom_config["block_size"],
        n_embd=custom_config["n_embd"],
        n_layer=custom_config["n_layer"],
        n_head=custom_config["n_head"],
        n_inner=4 * custom_config["n_embd"],
        activation_function="gelu",
        resid_pdrop=custom_config["dropout"],
        embd_pdrop=custom_config["dropout"],
        attn_pdrop=custom_config["dropout"],
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=custom_config.get("bos_token_id", 0),
        eos_token_id=custom_config.get("eos_token_id", 0),
        # HuggingFace GPT2 does NOT support bias=False out of the box in older versions, 
        # but modern transformers supports `add_cross_attention` etc. 
        # Actually NanoRushGPT might use bias=False. We'll manually inject if needed, 
        # or we just load it using a custom script. Wait, GPT2Config supports `layer_norm_bias`.
    )
    
    # Map weights
    print("Loading safetensors...")
    state_dict = load_file(f"{model_dir}/model.safetensors")
    
    hf_state_dict = {}
    
    # Strip _orig_mod prefix if it exists
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            clean_state_dict[k] = v
            
    state_dict = clean_state_dict

    print("Mapping weights to Hugging Face format...")
    # Handle custom quantization if present
    quantized_keys = [k for k in state_dict.keys() if k.endswith(".__scale__")]
    if quantized_keys:
        print(f"De-quantizing {len(quantized_keys)} tensors for HF compatibility (restoring to FP16)...")
        for scale_key in quantized_keys:
            base_key = scale_key.replace(".__scale__", "")
            if base_key in state_dict:
                scale = state_dict[scale_key].float()
                weight = state_dict[base_key].float()
                state_dict[base_key] = (weight * scale).to(torch.float16)
            del state_dict[scale_key]

    for k, v in state_dict.items():
        # Transpose Linear weights because HF GPT2 uses Conv1D (in_features, out_features)
        if any(x in k for x in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
            v = v.t()
            
        if k == "wte.weight":
            hf_state_dict["transformer.wte.weight"] = v
            hf_state_dict["lm_head.weight"] = v
        elif k == "wpe.weight":
            hf_state_dict["transformer.wpe.weight"] = v
        elif k.startswith("ln_f"):
            hf_state_dict[f"transformer.{k}"] = v
        elif k.startswith("h."):
            hf_state_dict[f"transformer.{k}"] = v
        elif k == "lm_head.weight":
            hf_state_dict["lm_head.weight"] = v
        else:
            hf_state_dict[k] = v

    print("Initializing HF GPT2LMHeadModel...")
    hf_model = GPT2LMHeadModel(hf_config)
    
    # Check if NanoRushGPT used bias=False, but HF GPT-2 expects biases
    # We will just inject zeros for missing biases so HF GPT-2 loads cleanly
    model_dict = hf_model.state_dict()
    for name, param in model_dict.items():
        if name not in hf_state_dict:
            if name.endswith(".bias"):
                print(f"  -> Injecting zero bias for {name} (NanoRushGPT was bias=False)")
                hf_state_dict[name] = torch.zeros_like(param)
            else:
                print(f"  -> Warning: missing {name}")

    print("Loading mapped weights into HF model...")
    hf_model.load_state_dict(hf_state_dict, strict=True)
    
    print(f"Saving Hugging Face model to {out_dir}...")
    hf_model = hf_model.half() # Convert back to FP16 to save 50% disk space
    hf_model.save_pretrained(out_dir, safe_serialization=True)
    
    print("Exporting Tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{model_dir}/tokenizer.json",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>"
    )
    tokenizer.save_pretrained(out_dir)
    
    print("\n✅ Success! Your model is now a standard Hugging Face model.")
    print(f"You can now load it via: AutoModelForCausalLM.from_pretrained('{out_dir}')")

if __name__ == "__main__":
    # Must import GPT2Config from transformers, aliased here to fix the conflict
    from transformers import GPT2Config
    GPTConfig = GPT2Config
    main()
