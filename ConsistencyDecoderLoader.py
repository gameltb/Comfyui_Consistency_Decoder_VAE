import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy import model_management

from .consistencydecoder import ConsistencyDecoder


class ConsistencyDecoderWrapper:
    def __init__(self, decoder, device=None):
        self.decoder = decoder.eval()

        if device is None:
            device = model_management.vae_device()

        self.device = device
        self.offload_device = model_management.vae_offload_device()
        self.vae_dtype = model_management.vae_dtype()
        self.decoder = self.decoder.to(self.vae_dtype)

    def decode(self, x):
        self.decoder = self.decoder.to(self.device)
        try:
            memory_used = (2562 * x.shape[2] * x.shape[3] * 64) * 1.7 * 3
            model_management.free_memory(memory_used, self.device)
            result = self.decoder(x.to(self.device,self.vae_dtype))
        finally:
            self.decoder = self.decoder.to(self.offload_device)
        return result


class ConsistencyDecoderVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"vae_name": (folder_paths.get_filename_list("vae"),), },
        }
    RETURN_TYPES = ("ConsistencyVAE",)
    FUNCTION = "load_consistency_decoder"

    CATEGORY = "loaders"

    def load_consistency_decoder(self, vae_name):
        model_path = folder_paths.get_full_path("vae", vae_name)
        consistencyDecoder = ConsistencyDecoder(model_path, device="cpu")
        vae = ConsistencyDecoderWrapper(consistencyDecoder)
        return (vae,)


class ConsistencyDecoderVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("ConsistencyVAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples):
        image = vae.decode(samples["samples"])
        image = image[0].cpu().numpy()
        image = (image + 1.0) * 127.5
        image = image.clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image.transpose(1, 2, 0))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, )


NODE_CLASS_MAPPINGS = {
    "ConsistencyDecoderVAELoader": ConsistencyDecoderVAELoader,
    "ConsistencyDecoderVAEDecode": ConsistencyDecoderVAEDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConsistencyDecoderVAELoader": "Consistency Decoder VAE Loader",
    "ConsistencyDecoderVAEDecode": "Consistency Decoder VAE Decode",
}
