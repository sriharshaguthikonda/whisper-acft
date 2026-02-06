#!/usr/bin/env python3
"""
Convert safetensors HF model to PyTorch .pt format for whisper.cpp conversion.

Usage:
    python convert_safetensors_to_pt.py --model_dir ./model_train-tiny3_extracted --output ./model.pt

    python whisper.cpp/models/convert-hf-to-gguf.py --hf-repo ./model_train-tiny3_extracted  --outdir ggml_output --output-format gguf --qtype q8_0
"""

import os
import sys
import json
import argparse

def convert_safetensors_to_pt(model_dir: str, output_path: str):
    """Convert HF safetensors model to OpenAI-style .pt checkpoint."""
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch safetensors")
        sys.exit(1)
    
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(safetensors_path):
        print(f"Error: {safetensors_path} not found")
        sys.exit(1)
    
    print(f"Loading safetensors from {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Map HF keys to OpenAI keys
    # HF format: model.encoder.layers.0.self_attn.k_proj.weight
    # OpenAI format: encoder.blocks.0.attn.key.weight
    
    new_state_dict = {}
    key_mapping = {
        "model.encoder.embed_positions.weight": "encoder.positional_embedding",
        "model.encoder.layer_norm.weight": "encoder.ln_post.weight",
        "model.encoder.layer_norm.bias": "encoder.ln_post.bias",
        "model.decoder.embed_positions.weight": "decoder.positional_embedding",
        "model.decoder.embed_tokens.weight": "decoder.token_embedding.weight",
        "model.decoder.layer_norm.weight": "decoder.ln.weight",
        "model.decoder.layer_norm.bias": "decoder.ln.bias",
        "proj_out.weight": "decoder.token_embedding.weight",  # tied weights
    }
    
    # Conv layers
    key_mapping["model.encoder.conv1.weight"] = "encoder.conv1.weight"
    key_mapping["model.encoder.conv1.bias"] = "encoder.conv1.bias"
    key_mapping["model.encoder.conv2.weight"] = "encoder.conv2.weight"
    key_mapping["model.encoder.conv2.bias"] = "encoder.conv2.bias"
    
    for old_key, tensor in state_dict.items():
        new_key = old_key
        
        # Direct mapping
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
        # Encoder layers
        elif "model.encoder.layers." in old_key:
            new_key = old_key.replace("model.encoder.layers.", "encoder.blocks.")
            new_key = new_key.replace(".self_attn.", ".attn.")
            new_key = new_key.replace(".self_attn_layer_norm.", ".attn_ln.")
            new_key = new_key.replace(".final_layer_norm.", ".mlp_ln.")
            new_key = new_key.replace(".fc1.", ".mlp.0.")
            new_key = new_key.replace(".fc2.", ".mlp.2.")
            new_key = new_key.replace(".q_proj.", ".query.")
            new_key = new_key.replace(".k_proj.", ".key.")
            new_key = new_key.replace(".v_proj.", ".value.")
            new_key = new_key.replace(".out_proj.", ".out.")
        # Decoder layers  
        elif "model.decoder.layers." in old_key:
            new_key = old_key.replace("model.decoder.layers.", "decoder.blocks.")
            new_key = new_key.replace(".self_attn.", ".attn.")
            new_key = new_key.replace(".self_attn_layer_norm.", ".attn_ln.")
            new_key = new_key.replace(".encoder_attn.", ".cross_attn.")
            new_key = new_key.replace(".encoder_attn_layer_norm.", ".cross_attn_ln.")
            new_key = new_key.replace(".final_layer_norm.", ".mlp_ln.")
            new_key = new_key.replace(".fc1.", ".mlp.0.")
            new_key = new_key.replace(".fc2.", ".mlp.2.")
            new_key = new_key.replace(".q_proj.", ".query.")
            new_key = new_key.replace(".k_proj.", ".key.")
            new_key = new_key.replace(".v_proj.", ".value.")
            new_key = new_key.replace(".out_proj.", ".out.")
        
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            print(f"  {old_key} -> {new_key}")
    
    # Create dims dict (OpenAI format)
    dims = {
        "n_mels": config.get("num_mel_bins", 80),
        "n_vocab": config.get("vocab_size", 51865),
        "n_audio_ctx": config.get("max_source_positions", 1500),
        "n_audio_state": config.get("d_model", 384),
        "n_audio_head": config.get("encoder_attention_heads", 6),
        "n_audio_layer": config.get("encoder_layers", 4),
        "n_text_ctx": config.get("max_target_positions", 448),
        "n_text_state": config.get("d_model", 384),
        "n_text_head": config.get("decoder_attention_heads", 6),
        "n_text_layer": config.get("decoder_layers", 4),
    }
    
    # Save in OpenAI format
    checkpoint = {
        "dims": dims,
        "model_state_dict": new_state_dict,
    }
    
    print(f"\nSaving to {output_path}...")
    torch.save(checkpoint, output_path)
    print(f"Done! Model saved with dims: {dims}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert safetensors to PyTorch .pt")
    parser.add_argument("--model_dir", type=str, required=True, help="HF model directory")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file path")
    args = parser.parse_args()
    
    convert_safetensors_to_pt(args.model_dir, args.output)

if __name__ == "__main__":
    main()
