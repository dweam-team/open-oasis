import os
import sys
from pydantic import Field
import torch
import pygame
import numpy as np
from dweam import Game
from pathlib import Path
from huggingface_hub import snapshot_download as hf_snapshot_download
from dweam import get_cache_dir
from safetensors.torch import load_file
from torchvision.io import read_video

from .dit import DiT_models
from .vae import VAE_models

from .utils import ACTION_KEYS, sigmoid_beta_schedule
from einops import rearrange


key_map = {
    pygame.K_e: "inventory",
    pygame.K_ESCAPE: "ESC",
    pygame.K_1: "hotbar.1",
    pygame.K_2: "hotbar.2",
    pygame.K_3: "hotbar.3",
    pygame.K_4: "hotbar.4",
    pygame.K_5: "hotbar.5",
    pygame.K_6: "hotbar.6",
    pygame.K_7: "hotbar.7",
    pygame.K_8: "hotbar.8",
    pygame.K_9: "hotbar.9",
    pygame.K_w: "forward",
    pygame.K_s: "back",
    pygame.K_a: "left",
    pygame.K_d: "right",
    pygame.K_SPACE: "jump",
    pygame.K_LSHIFT: "sneak",
    pygame.K_LCTRL: "sprint",
    pygame.K_q: "drop",
}

mouse_map = {
    pygame.BUTTON_LEFT: "attack",
    pygame.BUTTON_RIGHT: "use",
}


# Helper functions to capture live actions
def get_current_action(mouse_rel, keys_pressed, mouse_buttons):
    action = {}

    for key, action_key in key_map.items():
        if key in keys_pressed:
            action[action_key] = 1
        else:
            action[action_key] = 0

    for button, action_key in mouse_map.items():
        if button in mouse_buttons:
            action[action_key] = 1
        else:
            action[action_key] = 0

    # Map keys to actions
    action["camera"] = (mouse_rel[1] / 4, mouse_rel[0] / 4)  # tuple (x, y)
    action["swapHands"] = 0  # Map to a key if needed
    action["pickItem"] = 0  # Map to a key if needed

    return action


def action_to_tensor(action, device):
    actions_one_hot = torch.zeros(len(ACTION_KEYS), device=device)
    for j, action_key in enumerate(ACTION_KEYS):
        if action_key.startswith("camera"):
            if action_key == "cameraX":
                value = action["camera"][0]
            elif action_key == "cameraY":
                value = action["camera"][1]
            else:
                raise ValueError(f"Unknown camera action key: {action_key}")
            # Normalize value to be in [-1, 1]
            max_val = 20
            bin_size = 0.5
            num_buckets = int(max_val / bin_size)
            value = (value) / num_buckets
            value = max(min(value, 1.0), -1.0)
        else:
            value = action.get(action_key, 0)
            value = float(value)
        actions_one_hot[j] = value
    return actions_one_hot


@torch.inference_mode
def sample(x, actions_tensor, ddim_noise_steps, stabilization_level, alphas_cumprod, noise_range, noise_abs_max, model, device):
    """
    Sample function with constant alpha_next and stabilization_level implemented.

    Args:
        x (torch.Tensor): Current latent tensor of shape [B, T, C, H, W].
        actions_tensor (torch.Tensor): Actions tensor of shape [B, T, num_actions].
        ddim_noise_steps (int): Number of DDIM noise steps.
        stabilization_level (int): Level to stabilize the initial frames.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas for each timestep.
        noise_range (torch.Tensor): Noise schedule tensor.
        noise_abs_max (float): Maximum absolute noise value.
        model (torch.nn.Module): The diffusion model.

    Returns:
        torch.Tensor: Updated latent tensor after sampling.
    """
    B, context_length, C, H, W = x.shape

    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # Set up noise values
        t_ctx = torch.full((B, context_length - 1), stabilization_level - 1, dtype=torch.long, device=device)
        t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
        t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
        t_next = torch.where(t_next < 0, t, t_next)
        t = torch.cat([t_ctx, t], dim=1)
        t_next = torch.cat([t_ctx, t_next], dim=1)

        # Get model predictions
        with torch.autocast("cuda", dtype=torch.half):
            v = model(x, t, actions_tensor)

        # Compute x_start and x_noise
        x_start = alphas_cumprod[t].sqrt() * x - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

        # Compute alpha_next with constant values for context frames
        alpha_next = alphas_cumprod[t_next].clone()
        alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])

        # Ensure the last frame has alpha_next set to 1 if it's the first noise step
        if noise_idx == 1:
            alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

        # Compute the predicted x
        x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

        # Update only the last frame in the latent tensor
        x[:, -1:] = x_pred[:, -1:]

        # Optionally clamp the noise to maintain stability
        x[:, -1:] = torch.clamp(x[:, -1:], -noise_abs_max, noise_abs_max)

    return x


@torch.inference_mode
def encode(video, vae, n_prompt_frames, scaling_factor, device):
    x = video[:n_prompt_frames].unsqueeze(0).to(device)
    # VAE encoding
    x = rearrange(x, "b t h w c -> (b t) c h w").half()
    H, W = x.shape[-2:]
    with torch.no_grad():
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H//vae.patch_size, w=W//vae.patch_size)
    return x


@torch.inference_mode
def decode(x, vae, scaling_factor):
    # VAE decoding of the last frame
    x_last = x[:, -1:]
    x_last = rearrange(x_last, "b t c h w -> (b t) (h w) c").half()
    with torch.no_grad():
        x_decoded = (vae.decode(x_last / scaling_factor) + 1) / 2
    x_decoded = rearrange(x_decoded, "(b t) c h w -> b t h w c", b=1, t=1)
    x_decoded = torch.clamp(x_decoded, 0, 1)
    x_decoded = (x_decoded * 255).byte().cpu().numpy()
    frame = x_decoded[0, 0]
    return frame


def snapshot_download(**kwargs) -> Path:
    base_cache_dir = get_cache_dir()
    cache_dir = base_cache_dir / 'huggingface-data'
    path = hf_snapshot_download(cache_dir=str(cache_dir), **kwargs)
    return Path(path)


class OasisGame(Game):
    class Params(Game.Params):
        # context_window_size: int = Field(
        #     title="Context Window Size",
        #     default=4, description="Number of frames of memory"
        # )
        ddim_noise_steps: int = Field(
            title="Denoising Steps",
            default=16, description="Less steps means faster generation, but less stable"
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Device setup
        self.device = "cuda:0"
        assert torch.cuda.is_available()
        
        # Download model checkpoints
        path_hf = snapshot_download(
            repo_id="DeathDaDev/oasis-500m",
            allow_patterns=["oasis500m.safetensors", "vit-l-20.safetensors"]
        )
        self.model_path = path_hf / "oasis500m.safetensors"
        self.vae_path = path_hf / "vit-l-20.safetensors"
        
        # Sampling parameters
        self.B = 1
        self.max_noise_level = 1000
        # self.ddim_noise_steps = 16
        self.noise_abs_max = 20
        self.enable_torch_compile_model = True
        self.enable_torch_compile_vae = True
        
        # Context window parameters
        self.context_window_size = 4
        self.n_prompt_frames = 4
        self.offset = 0
        self.scaling_factor = 0.07843137255
        
        # Video parameters
        self.video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
        self.stabilization_level = 15
        
        # Display parameters
        self.screen_width = 640
        self.screen_height = 360
        
        # Load DiT model
        ckpt = load_file(self.model_path)
        self.model = DiT_models["DiT-S/2"]()
        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.to(self.device).half().eval()

        # Load VAE model
        vae_ckpt = load_file(self.vae_path)
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        self.vae.load_state_dict(vae_ckpt)
        self.vae = self.vae.to(self.device).half().eval()

        # Optional model compilation
        if self.enable_torch_compile_model:
            self.model = torch.compile(self.model, mode='reduce-overhead')
        if self.enable_torch_compile_vae:
            self.vae = torch.compile(self.vae, mode='reduce-overhead')

        # Setup noise schedule
        self.noise_range = torch.linspace(-1, self.max_noise_level - 1, self.params.ddim_noise_steps + 1).to(self.device)
        self.ctx_max_noise_idx = self.params.ddim_noise_steps // 10 * 3

        # Load video
        mp4_path = os.path.join(os.path.dirname(__file__), "sample_data", f"{self.video_id}.mp4")
        self.video = read_video(mp4_path, pts_unit="sec")[0].float() / 255

        self.video = self.video[self.offset:]

        self.x = None
        self.actions_list = []

        self.reset()
        
        # Get alphas
        betas = sigmoid_beta_schedule(self.max_noise_level).to(self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")
        
        # # Initialize with zero tensor for first frame
        # self.x = torch.randn((self.B, self.n_prompt_frames, 32, 32, 4), device=self.device)
        # self.x = torch.clamp(self.x, -self.noise_abs_max, self.noise_abs_max)
        
        # # Initialize actions list
        # self.actions_list = []
        # initial_action = torch.zeros(len(ACTION_KEYS), device=self.device).unsqueeze(0)
        # for _ in range(self.n_prompt_frames - 1):
        #     self.actions_list.append(initial_action)

        self.first_frame_generated = False

    def on_params_update(self, new_params: Params) -> None:
        super().on_params_update(new_params)
        self.noise_range = torch.linspace(-1, self.max_noise_level - 1, self.params.ddim_noise_steps + 1).to(self.device)
        self.ctx_max_noise_idx = self.params.ddim_noise_steps // 10 * 3

    def reset(self):
        self.x = encode(self.video, self.vae, self.n_prompt_frames, self.scaling_factor, self.device)
        # Initialize with initial action (assumed zero action)
        self.actions_list = []
        initial_action = torch.zeros(len(ACTION_KEYS), device=self.device).unsqueeze(0)
        for i in range(self.n_prompt_frames - 1):
            self.actions_list.append(initial_action)

    def step(self) -> pygame.Surface:
        if not self.first_frame_generated:
            print("Generating first frame, may take a while to warm up...", file=sys.stderr)
            self.first_frame_generated = True

        # Capture current action
        action = get_current_action(self.mouse_motion, self.keys_pressed, self.mouse_pressed)
        actions_curr = action_to_tensor(action, self.device).unsqueeze(0)
        self.actions_list.append(actions_curr)

        # Generate a random latent for the new frame
        chunk = torch.randn((self.B, 1, *self.x.shape[-3:]), device=self.device)
        chunk = torch.clamp(chunk, -self.noise_abs_max, self.noise_abs_max)
        self.x = torch.cat([self.x, chunk], dim=1)

        # Implement sliding window for context frames and actions
        if self.x.shape[1] > self.context_window_size:
            self.x = self.x[:, -self.context_window_size:]
            self.actions_list = self.actions_list[-self.context_window_size:]

        # Prepare actions tensor and sample
        actions_tensor = torch.stack(self.actions_list, dim=1)

        self.x = sample(
            self.x, actions_tensor, self.params.ddim_noise_steps, 
            self.stabilization_level, self.alphas_cumprod, 
            self.noise_range, self.noise_abs_max, self.model, self.device
        )

        # Decode frame
        frame = decode(self.x, self.vae, self.scaling_factor)

        # Convert to pygame surface
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        return pygame.transform.scale(frame_surface, (self.screen_width, self.screen_height))

    def on_key_down(self, key: int) -> None:
        if key == pygame.K_RETURN:
            self.reset()

    def stop(self) -> None:
        super().stop()
        # Clean up resources
        self.model = None
        self.vae = None
        torch.cuda.empty_cache()
