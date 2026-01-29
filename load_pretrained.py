import torch
from transformers import GPT2LMHeadModel
from model.gpt1 import GPT1Model
from config import GPT1Config
import os

def convert_gpt2_to_gpt1(gpt2_model_name='gpt2'):
    print(f"Loading pretrained GPT-2 model: {gpt2_model_name}")
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    gpt2_state = gpt2_model.state_dict()
    
    config = GPT1Config()
    config.vocab_size = gpt2_model.config.vocab_size
    config.max_seq_len = gpt2_model.config.n_positions
    config.d_model = gpt2_model.config.n_embd
    config.n_layers = gpt2_model.config.n_layer
    config.n_heads = gpt2_model.config.n_head
    config.d_ff = gpt2_model.config.n_inner if gpt2_model.config.n_inner else 4 * config.d_model
    
    print("Creating GPT-1 model with GPT-2 architecture...")
    gpt1_model = GPT1Model(config)
    gpt1_state = gpt1_model.state_dict()
    
    print("Transferring weights...")
    
    mapping = {
        'transformer.wte.weight': 'token_embedding.weight',
        'transformer.wpe.weight': 'pos_encoding.pe',
        'transformer.ln_f.weight': 'ln_f.weight',
        'transformer.ln_f.bias': 'ln_f.bias',
        'lm_head.weight': 'lm_head.weight',
    }
    
    transferred = {}
    
    for gpt2_key, gpt1_key in mapping.items():
        if gpt2_key in gpt2_state and gpt1_key in gpt1_state:
            if 'pos_encoding' in gpt1_key:
                transferred[gpt1_key] = gpt2_state[gpt2_key].unsqueeze(0)
            else:
                transferred[gpt1_key] = gpt2_state[gpt2_key]
    
    for i in range(config.n_layers):
        block_mappings = {
            f'transformer.h.{i}.ln_1.weight': f'blocks.{i}.ln1.weight',
            f'transformer.h.{i}.ln_1.bias': f'blocks.{i}.ln1.bias',
            f'transformer.h.{i}.ln_2.weight': f'blocks.{i}.ln2.weight',
            f'transformer.h.{i}.ln_2.bias': f'blocks.{i}.ln2.bias',
            f'transformer.h.{i}.mlp.c_fc.weight': f'blocks.{i}.feed_forward.linear1.weight',
            f'transformer.h.{i}.mlp.c_fc.bias': f'blocks.{i}.feed_forward.linear1.bias',
            f'transformer.h.{i}.mlp.c_proj.weight': f'blocks.{i}.feed_forward.linear2.weight',
            f'transformer.h.{i}.mlp.c_proj.bias': f'blocks.{i}.feed_forward.linear2.bias',
        }
        
        for gpt2_key, gpt1_key in block_mappings.items():
            if gpt2_key in gpt2_state and gpt1_key in gpt1_state:
                weight = gpt2_state[gpt2_key]
                if 'weight' in gpt2_key and len(weight.shape) == 2:
                    weight = weight.T
                transferred[gpt1_key] = weight
        
        attn_weight = gpt2_state[f'transformer.h.{i}.attn.c_attn.weight'].T
        attn_bias = gpt2_state[f'transformer.h.{i}.attn.c_attn.bias']
        
        d_model = config.d_model
        
        q_weight, k_weight, v_weight = attn_weight.split(d_model, dim=0)
        q_bias, k_bias, v_bias = attn_bias.split(d_model, dim=0)
        
        transferred[f'blocks.{i}.attention.q_linear.weight'] = q_weight
        transferred[f'blocks.{i}.attention.q_linear.bias'] = q_bias
        transferred[f'blocks.{i}.attention.k_linear.weight'] = k_weight
        transferred[f'blocks.{i}.attention.k_linear.bias'] = k_bias
        transferred[f'blocks.{i}.attention.v_linear.weight'] = v_weight
        transferred[f'blocks.{i}.attention.v_linear.bias'] = v_bias
        
        proj_weight = gpt2_state[f'transformer.h.{i}.attn.c_proj.weight'].T
        proj_bias = gpt2_state[f'transformer.h.{i}.attn.c_proj.bias']
        transferred[f'blocks.{i}.attention.out_proj.weight'] = proj_weight
        transferred[f'blocks.{i}.attention.out_proj.bias'] = proj_bias
    
    gpt1_state.update(transferred)
    gpt1_model.load_state_dict(gpt1_state, strict=False)
    
    print(f"Weight transfer complete!")
    print(f"Transferred {len(transferred)} parameter tensors")
    
    return gpt1_model, config


def save_pretrained_checkpoint(output_path='checkpoints/pretrained_gpt2.pt', gpt2_model_name='gpt2'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model, config = convert_gpt2_to_gpt1(gpt2_model_name)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'global_step': 0,
        'epoch': 0,
    }
    
    torch.save(checkpoint, output_path)
    print(f"Pretrained checkpoint saved to: {output_path}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load pretrained GPT-2 weights into GPT-1 model')
    parser.add_argument('--model', type=str, default='gpt2', help='GPT-2 model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--output', type=str, default='checkpoints/pretrained_gpt2.pt', help='Output checkpoint path')
    args = parser.parse_args()
    
    save_pretrained_checkpoint(args.output, args.model)
