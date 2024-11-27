import argparse
import torch
from open_clip import create_model_and_transforms
from transformers import CLIPModel, CLIPConfig
from safetensors.torch import save_file

def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    # Split `in_proj_weight` into query, key, and value projections
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    # Map to Hugging Face attention layers
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias
    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias
    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias

    # Output projection layer
    hf_attn_layer.out_proj.weight.data = pt_attn_layer.out_proj.weight
    hf_attn_layer.out_proj.bias.data = pt_attn_layer.out_proj.bias


def copy_mlp(hf_mlp, pt_mlp):
    # Copy fully connected layers (c_fc -> fc1, c_proj -> fc2)
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    hf_linear.weight.data = pt_linear.weight
    hf_linear.bias.data = pt_linear.bias


def copy_layer(hf_layer, pt_layer):
    # Layer norms
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # MLP and attention components
    copy_mlp(hf_layer.mlp, pt_layer.mlp)
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model, clip_skip=0):
    # Token embeddings and positional embeddings
    hf_encoder.embeddings.token_embedding.weight.data = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding

    # Final layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # Handle clip_skip by slicing the transformer layers
    resblocks = pt_model.transformer.resblocks
    layers_to_copy = resblocks[:-clip_skip] if clip_skip > 0 else resblocks
    copy_layers(hf_encoder.encoder.layers, layers_to_copy)


def copy_text_model_and_projection(hf_model, pt_model, clip_skip=0):
    hf_model.text_projection.weight.data = pt_model.text_projection.T
    copy_encoder(hf_model.text_model, pt_model, clip_skip=clip_skip)


def copy_vision_model_and_projection(hf_model, pt_model):
    # Visual projection weights
    hf_model.visual_projection.weight.data = pt_model.visual.proj.T

    # Layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # Embedding layers
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding
    hf_model.vision_model.embeddings.class_embedding.data = pt_model.visual.class_embedding

    # Transformer layers
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)



@torch.no_grad()
def convert_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, model_name, clip_skip=0, config_path=None):
    # Load the OpenCLIP model
    pt_model, _, _ = create_model_and_transforms(model_name, pretrained=checkpoint_path, device="cpu")
    pt_model.eval()

    # Generate the Hugging Face configuration
    if config_path is not None:
        config = CLIPConfig.from_pretrained(config_path)
    else:
        config = CLIPConfig(
            text_config={
                "vocab_size": pt_model.vocab_size,
                "hidden_size": pt_model.text_projection.shape[0],
            },
            vision_config={
                "hidden_size": pt_model.visual.proj.shape[0],
            },
            projection_dim=pt_model.text_projection.shape[0],
        )

    # Create Hugging Face model
    hf_model = CLIPModel(config).eval()

    # Copy weights
    copy_text_model_and_projection(hf_model, pt_model, clip_skip=clip_skip)
    copy_vision_model_and_projection(hf_model, pt_model)

    # Copy logit scale
    hf_model.logit_scale.data = pt_model.logit_scale.data

    # Extract the full state_dict
    state_dict = hf_model.state_dict()

    # Ensure all tensors are contiguous
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and not value.is_contiguous():
            state_dict[key] = value.contiguous()

    # Save the entire state_dict into a single safetensors file
    output_file = f"{pytorch_dump_folder_path}/model.safetensors"
    save_file(state_dict, output_file, metadata={"format": "pt"})
    print(f"Model saved as a single file at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ViT-bigG-14")
    parser.add_argument("--clip_skip", type=int, default=0, help="Number of encoder layers to skip.")
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    convert_clip_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.model_name,
        clip_skip=args.clip_skip,
        config_path=args.config_path,
    )
