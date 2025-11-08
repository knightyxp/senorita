import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from control_cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from control_cogvideox.controlnet_cogvideox_transformer_3d import ControlCogVideoXTransformer3DModel
from pipeline_cogvideox_controlnet_5b_i2v_instruction2 import ControlCogVideoXPipeline


@dataclass
class TaskItem:
    stem: str
    source_video_path: str
    prompt: str
    negative_prompt: str
    guidance_scale: float


def _align_to_multiple(value: int, base: int = 8) -> int:
    if value % base == 0:
        return value
    return max(base, math.ceil(value / base) * base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed inference for ControlCogVideoXPipeline with first-frame conditioning."
    )
    parser.add_argument("--tasks_json", type=str, required=True, help="Path to JSON file describing tasks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store generated results.")
    parser.add_argument(
        "--model_root",
        type=str,
        default="./cogvideox-5b-i2v",
        help="Root directory containing CogVideoX 5B I2V model weights.",
    )
    parser.add_argument(
        "--control_checkpoint",
        type=str,
        default="./senorita-2m/models_half/ff_controlnet_half.pth",
        help="Path to SENORITA controlnet checkpoint.",
    )
    parser.add_argument("--num_frames", type=int, default=33, help="Number of frames to generate (<= 49).")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height override. Defaults to aligned input height if not set.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width override. Defaults to aligned input width if not set.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for exported videos.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--default_prompt",
        type=str,
        default="",
        help="Fallback positive prompt when task JSON does not provide one.",
    )
    parser.add_argument(
        "--default_negative_prompt",
        type=str,
        default="",
        help="Fallback negative prompt when task JSON does not provide one.",
    )
    parser.add_argument(
        "--control_num_layers",
        type=int,
        default=6,
        help="Number of control transformer layers to instantiate.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip items that already have generated videos in output directory.",
    )
    return parser.parse_args()


def init_distributed() -> Tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def unwarp_model(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key.split("module.", 1)[1]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_pipeline(args: argparse.Namespace, device: torch.device, rank: int) -> ControlCogVideoXPipeline:
    i2v_key = "i2v"
    scheduler_cfg_path = os.path.join(args.model_root, f"cogvideox-5b-{i2v_key}", "scheduler", "scheduler_config.json")
    if not os.path.exists(scheduler_cfg_path):
        scheduler_cfg_path = os.path.join(args.model_root, "scheduler", "scheduler_config.json")

    scheduler_config = OmegaConf.to_container(OmegaConf.load(scheduler_cfg_path))
    scheduler = CogVideoXDDIMScheduler(**scheduler_config)

    model_root_candidates = [
        os.path.join(args.model_root, f"cogvideox-5b-{i2v_key}"),
        args.model_root,
    ]

    base_root = None
    for candidate in model_root_candidates:
        if os.path.isdir(candidate):
            base_root = candidate
            break
    if base_root is None:
        raise FileNotFoundError(f"Could not locate CogVideoX model root under '{args.model_root}'.")

    text_encoder = T5EncoderModel.from_pretrained(base_root, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(base_root, subfolder="vae", torch_dtype=torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(base_root, "tokenizer"))

    transformer_config_path = os.path.join(base_root, "transformer", "config.json")
    transformer_config = OmegaConf.to_container(OmegaConf.load(transformer_config_path))
    transformer_config["in_channels"] = 32
    transformer = CogVideoXTransformer3DModel(**transformer_config)

    control_config = OmegaConf.to_container(OmegaConf.load(transformer_config_path))
    control_config["in_channels"] = 32
    control_config["num_layers"] = args.control_num_layers
    control_config["control_in_channels"] = 16
    controlnet_transformer = ControlCogVideoXTransformer3DModel(**control_config)

    if not os.path.exists(args.control_checkpoint):
        raise FileNotFoundError(f"Control checkpoint not found at '{args.control_checkpoint}'.")
    all_state_dicts = torch.load(args.control_checkpoint, map_location="cpu")
    transformer.load_state_dict(unwarp_model(all_state_dicts["transformer_state_dict"]), strict=True)
    controlnet_transformer.load_state_dict(unwarp_model(all_state_dicts["controlnet_transformer_state_dict"]), strict=True)

    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    transformer = transformer.to(device).half()
    controlnet_transformer = controlnet_transformer.to(device).half()

    pipe = ControlCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder.half(),
        vae=vae.half(),
        transformer=transformer,
        scheduler=scheduler,
        controlnet_transformer=controlnet_transformer,
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=rank != 0)
    pipe.to(device)
    pipe._execution_device = device  # noqa: SLF001

    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.transformer.eval()
    pipe.controlnet_transformer.eval()

    return pipe


def derive_ground_instruction(edit_instruction_text: str) -> str:
    s = (edit_instruction_text or "").strip()
    if s.endswith("."):
        s = s[:-1]
    lower = s.lower()
    for prefix in ["remove ", "delete ", "erase ", "eliminate ", "add ", "make ", "ground "]:
        if lower.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s


def resolve_prompts(item: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str, float]:
    prompt = (
        item.get("prompt")
        or item.get("positive_prompt")
        or item.get("qwen_vl_72b_refined_instruction")
        or item.get("edit_instruction")
        or item.get("text")
        or args.default_prompt
    )
    negative = item.get("negative_prompt", args.default_negative_prompt)

    if args.videoedit_reasoning:
        ground = derive_ground_instruction(prompt)
        prompt = (
            "A video sequence showing three parts: first the original scene, "
            f"then grounded {ground}, and finally the same scene but {prompt}"
        )
    else:
        prompt = (
            "A video sequence showing two parts: the first half shows the original scene, "
            f"and the second half shows the same scene but {prompt}"
        )

    guidance_scale = float(item.get("guidance_scale", args.guidance_scale))

    return prompt, negative, guidance_scale


def infer_output_stem(item: Dict[str, Any]) -> str:
    if "output_name" in item and item["output_name"]:
        return os.path.splitext(os.path.basename(item["output_name"]))[0]
    if "task_type" in item and "sample_id" in item:
        return f"{item['task_type']}_{item['sample_id']}"
    if "id" in item:
        return str(item["id"])
    if "source_video_path" in item and item["source_video_path"]:
        return os.path.splitext(os.path.basename(item["source_video_path"]))[0]
    raise ValueError("Unable to infer output stem from task item.")


def build_tasks(args: argparse.Namespace) -> List[TaskItem]:
    with open(args.tasks_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items_iter = data.items()
    else:
        items_iter = enumerate(data)

    tasks: List[TaskItem] = []
    for _, raw_item in items_iter:
        if "source_video_path" not in raw_item:
            raise KeyError("Each task must include 'source_video_path'.")

        prompt, negative, guidance = resolve_prompts(raw_item, args)
        stem = infer_output_stem(raw_item)
        tasks.append(
            TaskItem(
                stem=stem,
                source_video_path=raw_item["source_video_path"],
                prompt=prompt,
                negative_prompt=negative,
                guidance_scale=guidance,
            )
        )
    return tasks


def split_work(tasks: List[TaskItem], rank: int, world_size: int) -> List[TaskItem]:
    if world_size <= 1:
        return tasks
    return tasks[rank::world_size]


def load_video_frames(
    video_path: str,
    num_frames: int,
    target_height: Optional[int],
    target_width: Optional[int],
) -> Tuple[np.ndarray, int, int]:
    reader = imageio.get_reader(video_path)
    raw_frames: List[np.ndarray] = []
    for idx in range(num_frames):
        try:
            frame = reader.get_data(idx)
        except IndexError:
            break
        raw_frames.append(np.asarray(frame, dtype=np.uint8))
    reader.close()

    if not raw_frames:
        raise RuntimeError(f"Failed to read any frames from '{video_path}'.")

    original_height, original_width = raw_frames[0].shape[0], raw_frames[0].shape[1]

    final_height = _align_to_multiple(target_height if target_height is not None else original_height)
    final_width = _align_to_multiple(target_width if target_width is not None else original_width)

    if target_height is not None and final_height != target_height:
        print(f"[WARN] Requested height {target_height} adjusted to {final_height} (must be divisible by 8).")
    if target_width is not None and final_width != target_width:
        print(f"[WARN] Requested width {target_width} adjusted to {final_width} (must be divisible by 8).")

    if final_height != original_height or final_width != original_width:
        print(
            f"Resizing frames from {original_width}x{original_height} to {final_width}x{final_height} "
            f"for '{os.path.basename(video_path)}'."
        )

    frames: List[np.ndarray] = []
    for frame in raw_frames:
        pil_frame = Image.fromarray(frame, mode="RGB")
        if pil_frame.size != (final_width, final_height):
            pil_frame = pil_frame.resize((final_width, final_height), Image.BICUBIC)
        frames.append(np.array(pil_frame, dtype=np.uint8))

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    frames_np = np.stack(frames, axis=0)
    return frames_np, final_height, final_width


def prepare_conditions(
    pipe: ControlCogVideoXPipeline,
    frames_np: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    video = torch.from_numpy(frames_np).to(device=device, dtype=torch.float32)
    video = video.unsqueeze(0)  # [1, F, H, W, C]

    source_pixel_values = video / 127.5 - 1.0
    source_pixel_values = source_pixel_values.permute(0, 4, 1, 2, 3).contiguous()

    with torch.no_grad():
        source_latents = pipe.vae.encode(source_pixel_values.to(dtype=torch.float16)).latent_dist.sample()
        source_latents = source_latents * pipe.vae.config.scaling_factor
    source_latents = source_latents.permute(0, 2, 1, 3, 4).contiguous()

    first_frame = video[:, :1]
    target_pixel_values = first_frame / 127.5 - 1.0
    target_pixel_values = target_pixel_values.permute(0, 4, 1, 2, 3).contiguous()

    with torch.no_grad():
        image_latents = pipe.vae.encode(target_pixel_values.to(dtype=torch.float16)).latent_dist.sample()
        image_latents = image_latents * pipe.vae.config.scaling_factor
    image_latents = image_latents.permute(0, 2, 1, 3, 4).contiguous()

    if image_latents.shape[1] < source_latents.shape[1]:
        pad = torch.zeros(
            image_latents.shape[0],
            source_latents.shape[1] - image_latents.shape[1],
            image_latents.shape[2],
            image_latents.shape[3],
            image_latents.shape[4],
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        image_latents = torch.cat([image_latents, pad], dim=1)
    elif image_latents.shape[1] > source_latents.shape[1]:
        image_latents = image_latents[:, : source_latents.shape[1]]

    return source_latents.to(dtype=torch.float16), image_latents.to(dtype=torch.float16)


def save_input_video(frames_np: np.ndarray, file_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames_np]
    export_to_video(pil_frames, file_path, fps=fps)
    print(f"Saved input video → {file_path}")


def save_generated_video(frames: List[Image.Image], file_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    export_to_video(frames, file_path, fps=fps)
    print(f"Saved generated video → {file_path}")


def save_side_by_side(
    source_frames_np: np.ndarray,
    generated_frames: List[Image.Image],
    file_path: str,
    fps: int,
) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    length = min(len(source_frames_np), len(generated_frames))
    side_by_side_frames: List[Image.Image] = []
    for idx in range(length):
        src = Image.fromarray(source_frames_np[idx].astype(np.uint8)).convert("RGB")
        gen = generated_frames[idx]
        if gen.size != src.size:
            gen = gen.resize(src.size, Image.BICUBIC)
        combined = Image.new("RGB", (src.width + gen.width, src.height))
        combined.paste(src, (0, 0))
        combined.paste(gen, (src.width, 0))
        side_by_side_frames.append(combined)
    export_to_video(side_by_side_frames, file_path, fps=fps)
    print(f"Saved side-by-side video → {file_path}")


def run_task(
    pipe: ControlCogVideoXPipeline,
    task: TaskItem,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    frames_np, video_height, video_width = load_video_frames(
        task.source_video_path, args.num_frames, args.height, args.width
    )
    source_latents, image_latents = prepare_conditions(pipe, frames_np, device)

    output = pipe(
        prompt=task.prompt,
        negative_prompt=task.negative_prompt,
        video_condition=source_latents,
        video_condition2=image_latents,
        height=video_height,
        width=video_width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=task.guidance_scale,
        generator=generator,
        output_type="pil",
        return_dict=True,
    )
    generated_frames = output.frames[0]

    base = os.path.join(args.output_dir, task.stem)
    input_path = f"{base}_input.mp4"
    gen_path = f"{base}_gen.mp4"
    compare_path = f"{base}_compare.mp4"
    info_path = f"{base}_prompt.txt"

    save_input_video(frames_np, input_path, args.fps)
    save_generated_video(generated_frames, gen_path, args.fps)
    save_side_by_side(frames_np, generated_frames, compare_path, args.fps)

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(task.prompt)

    print(f"[Done] Saved outputs for stem '{task.stem}'.")


def filter_pending(tasks: List[TaskItem], args: argparse.Namespace) -> List[TaskItem]:
    if not args.skip_existing:
        return tasks
    pending: List[TaskItem] = []
    for task in tasks:
        output_path = os.path.join(args.output_dir, f"{task.stem}_gen.mp4")
        if not os.path.exists(output_path):
            pending.append(task)
    return pending


def main() -> None:
    args = parse_args()

    if args.num_frames > 49:
        raise ValueError("num_frames must be <= 49 due to rotary embedding limitation.")

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for parallel execution.")

    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Starting distributed inference with {world_size} GPUs.")

    pipe = load_pipeline(args, device, rank)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    all_tasks = build_tasks(args)
    pending_tasks = filter_pending(all_tasks, args)

    if rank == 0:
        print(
            f"Total tasks: {len(all_tasks)}, pending: {len(pending_tasks)}, skipped: {len(all_tasks) - len(pending_tasks)}"
        )

    local_tasks = split_work(pending_tasks, rank, world_size)
    print(f"[GPU {rank}] Assigned {len(local_tasks)} tasks.")

    for task in local_tasks:
        try:
            print(f"[GPU {rank}] Processing {task.stem} ...")
            run_task(pipe, task, args, device, generator)
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"[GPU {rank}] Error while processing '{task.stem}': {exc}")

    dist.barrier()
    if rank == 0:
        print("All tasks completed.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

