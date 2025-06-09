import logging
import copy
import folder_paths
from comfy.comfy_types import IO # type: ignore
import asyncio
from .utils import FlexibleOptionalInputType, any_type, get_lora_info, extract_lora_name, get_loras_list
from nunchaku.lora.flux import to_diffusers

logger = logging.getLogger(__name__)

def load_lora(model, lora_name: str, lora_strength: float):
    if abs(lora_strength) < 1e-5:
        return (model)
    model_wrapper = model.model.diffusion_model
    # assert isinstance(model_wrapper, ComfyFluxWrapper)

    transformer = model_wrapper.model
    model_wrapper.model = None
    ret_model = copy.deepcopy(model)  # copy everything except the model
    ret_model_wrapper = ret_model.model.diffusion_model
    # assert isinstance(ret_model_wrapper, ComfyFluxWrapper)

    model_wrapper.model = transformer
    ret_model_wrapper.model = transformer

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    ret_model_wrapper.loras.append((lora_path, lora_strength))

    sd = to_diffusers(lora_path)

    if "transformer.x_embedder.lora_A.weight" in sd:
        new_in_channels = sd["transformer.x_embedder.lora_A.weight"].shape[1]
        assert new_in_channels % 4 == 0
        new_in_channels = new_in_channels // 4

        old_in_channels = ret_model.model.model_config.unet_config["in_channels"]
        if old_in_channels < new_in_channels:
            ret_model.model.model_config.unet_config["in_channels"] = new_in_channels

    return (ret_model)

class LoraManagerLoader:
    NAME = "Lora Loader (LoraManager)"
    CATEGORY = "Lora Manager/loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": (IO.STRING, {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "tooltip": "Format: <lora:lora_name:strength> separated by spaces or punctuation",
                    "placeholder": "LoRA syntax input: <lora:name:strength>"
                }),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    RETURN_TYPES = ("MODEL", "CLIP", IO.STRING, IO.STRING)
    RETURN_NAMES = ("MODEL", "CLIP", "trigger_words", "loaded_loras")
    FUNCTION = "load_loras"
    
    def load_loras(self, model, text, **kwargs):
        """Loads multiple LoRAs based on the kwargs input and lora_stack."""
        loaded_loras = []
        all_trigger_words = []
        
        clip = kwargs.get('clip', None)
        # First process lora_stack if available
        # if lora_stack:
        #     for lora_path, model_strength, clip_strength in lora_stack:
        #         # Apply the LoRA using the provided path and strengths
        #         model, clip = LoraLoader().load_lora(model, clip, lora_path, model_strength, clip_strength)
                
        #         # Extract lora name for trigger words lookup
        #         lora_name = extract_lora_name(lora_path)
        #         _, trigger_words = asyncio.run(get_lora_info(lora_name))
                
        #         all_trigger_words.extend(trigger_words)
        #         # Add clip strength to output if different from model strength
        #         if abs(model_strength - clip_strength) > 0.001:
        #             loaded_loras.append(f"{lora_name}: {model_strength},{clip_strength}")
        #         else:
        #             loaded_loras.append(f"{lora_name}: {model_strength}")
        
        # Then process loras from kwargs with support for both old and new formats
        loras_list = get_loras_list(kwargs)
        for lora in loras_list:
            if not lora.get('active', False):
                continue
                
            lora_name = lora['name']
            model_strength = float(lora['strength'])
            # Get clip strength - use model strength as default if not specified
            clip_strength = float(lora.get('clipStrength', model_strength))
            
            # Get lora path and trigger words
            lora_path, trigger_words = asyncio.run(get_lora_info(lora_name))
            
            # Apply the LoRA using the resolved path with separate strengths
            model = load_lora(model, lora_path, model_strength)
            
            # Include clip strength in output if different from model strength
            if abs(model_strength - clip_strength) > 0.001:
                loaded_loras.append(f"{lora_name}: {model_strength},{clip_strength}")
            else:
                loaded_loras.append(f"{lora_name}: {model_strength}")
            
            # Add trigger words to collection
            all_trigger_words.extend(trigger_words)
        
        # use ',, ' to separate trigger words for group mode
        trigger_words_text = ",, ".join(all_trigger_words) if all_trigger_words else ""
        
        # Format loaded_loras with support for both formats
        formatted_loras = []
        for item in loaded_loras:
            parts = item.split(":")
            lora_name = parts[0].strip()
            strength_parts = parts[1].strip().split(",")
            
            if len(strength_parts) > 1:
                # Different model and clip strengths
                model_str = strength_parts[0].strip()
                clip_str = strength_parts[1].strip()
                formatted_loras.append(f"<lora:{lora_name}:{model_str}:{clip_str}>")
            else:
                # Same strength for both
                model_str = strength_parts[0].strip()
                formatted_loras.append(f"<lora:{lora_name}:{model_str}>")
                
        formatted_loras_text = " ".join(formatted_loras)

        return (model, clip, trigger_words_text, formatted_loras_text)