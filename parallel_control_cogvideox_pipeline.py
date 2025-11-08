#!/usr/bin/env python3
"""
Distributed 4-GPU inference script for the SENORITA Control CogVideoX pipeline.

This script loads editing tasks from a JSON file and runs them in parallel across
multiple GPUs using torch.distributed. The saved outputs follow the same naming
convention as `VideoX-Fun/examples/wan2.1/predict_v2v_json_new.py`, producing:

- `<base>_input.mp4`   : the conditioning video used by the pipeline
- `<base>_gen.mp4`     : the generated result video
- `<base>_compare.mp4` : side-by-side comparison of input | generated frames
- `<base>_gen_info.txt`: text file storing the prompt that was used

Launch with:
    torchrun --nproc_per_node=4 parallel_control_cogvideox_pipeline.py \
        --tasks_json path/to/tasks.json \
        --output_dir path/to/results \
        --model_root path/to/cogvideox-5b-i2v \
        --control_checkpoint path/to/ff_controlnet_half.pth
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import decord
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from diffusers import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer

from control_cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from control_cogvideox.controlnet_cogvideox_transformer_3d import ControlCogVideoXTransformer3DModel
from pipeline_cogvideox_controlnet_5b_i2v_instruction2 import ControlCogVideoXPipeline


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)


@dataclass
class TaskItem:
    base_name: str
    data: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SENORITA Control CogVideoX pipeline with 4-GPU distributed inference."
    )
    parser.add_argument("--tasks_json", type=str, required=True, help="Path to tasks JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store generated results.")
    parser.add_argument("--model_root", type=str, default="./cogvideox-5b-i2v", help="CogVideoX model root folder.")
    parser.add_argument(
        "--control_checkpoint",
        type=str,
        default="./senorita-2m/models_half/ff_controlnet_half.pth",
        help="Checkpoint containing transformer and controlnet weights.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--num_frames", type=int, default=33, help="Total frames to generate per sample.")
    parser.add_argument("--source_frames", type=int, default=33, help="Frames taken from source video per pass.")
    parser.add_argument("--height", type=int, default=448, help="Frame height used during generation.")
    parser.add_argument("--width", type=int, default=768, help="Frame width used during generation.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps per generation pass.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for exported videos.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Default negative prompt if a task does not provide one.",
    )
    parser.add_argument(
        "--control_num_layers",
        type=int,
        default=6,
        help="Number of control layers to enable in the control transformer.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose `<base>_gen.mp4` already exists in the output directory.",
    )
    parser.add_argument(
        "--edited_videos_dir",
        type=str,
        default=None,
        help="Directory containing generated videos named as `gen_<item_id>.mp4`; "
        "the script will use their first frames as reference images when available.",
    )
    return parser.parse_args()


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(base_dir, expanded))


def unwarp_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key.split("module.", 1)[1]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def init_pipeline(
    model_root: str,
    control_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    control_num_layers: int = 6,
) -> ControlCogVideoXPipeline:
    key = "i2v"
    scheduler_config = OmegaConf.to_container(
        OmegaConf.load(os.path.join(model_root, "scheduler", "scheduler_config.json"))
    )
    noise_scheduler = CogVideoXDDIMScheduler(**scheduler_config)

    text_encoder = T5EncoderModel.from_pretrained(
        model_root, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(model_root, subfolder="vae", torch_dtype=dtype)
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(model_root, "tokenizer"), torch_dtype=dtype)

    transformer_config = OmegaConf.to_container(OmegaConf.load(os.path.join(model_root, "transformer", "config.json")))
    transformer_config["in_channels"] = 32 if key == "i2v" else 16
    transformer = CogVideoXTransformer3DModel(**transformer_config)

    control_config = OmegaConf.to_container(OmegaConf.load(os.path.join(model_root, "transformer", "config.json")))
    control_config["in_channels"] = 32 if key == "i2v" else 16
    control_config["num_layers"] = control_num_layers
    control_config["control_in_channels"] = 16
    controlnet_transformer = ControlCogVideoXTransformer3DModel(**control_config)

    try:
        checkpoint = torch.load(control_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(control_checkpoint, map_location="cpu")
    transformer_state_dict = unwarp_model(checkpoint["transformer_state_dict"])
    controlnet_state_dict = unwarp_model(checkpoint["controlnet_transformer_state_dict"])
    transformer.load_state_dict(transformer_state_dict, strict=True)
    controlnet_transformer.load_state_dict(controlnet_state_dict, strict=True)

    transformer = transformer.to(dtype).eval()
    controlnet_transformer = controlnet_transformer.to(dtype).eval()
    vae = vae.eval()
    text_encoder = text_encoder.eval()

    pipe = ControlCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=noise_scheduler,
        controlnet_transformer=controlnet_transformer,
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(device=device)

    return pipe


def load_video_frames(
    video_path: str,
    target_frames: int,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, List[Image.Image], List[Image.Image], Tuple[int, int]]:
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    frame_indices = list(range(min(total_frames, target_frames)))
    frames_np = vr.get_batch(frame_indices).asnumpy()
    original_height, original_width = frames_np.shape[1], frames_np.shape[2]

    frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
    while len(frames_list) < target_frames:
        frames_list.append(frames_list[-1].copy())

    resized_frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC) for frame in frames_list]
    resized_np = np.stack(resized_frames, axis=0).astype(np.uint8)

    source_tensor = torch.from_numpy(resized_np).contiguous().unsqueeze(0)

    original_pil = [Image.fromarray(frame) for frame in frames_list]
    resized_pil = [Image.fromarray(frame) for frame in resized_frames]
    return source_tensor, original_pil, resized_pil, (original_height, original_width)


def load_video_frame_tensor(
    video_path: str,
    width: int,
    height: int,
    frame_index: int = 0,
) -> Optional[torch.Tensor]:
    if video_path is None or not os.path.exists(video_path):
        return None
    try:
        reader = decord.VideoReader(video_path)
        total = len(reader)
        if total == 0:
            return None
        index = frame_index if frame_index < total else total - 1
        frame = reader[index].asnumpy()
    except Exception as exc:
        print(f"Warning: unable to read frame {frame_index} from {video_path}: {exc}")
        return None

    try:
        image = Image.fromarray(frame)
    except Exception as exc:
        print(f"Warning: unable to convert frame {frame_index} from {video_path} to image: {exc}")
        return None

    resized = image.resize((width, height), Image.BICUBIC)
    arr = np.array(resized, dtype=np.uint8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def prepare_reference_tensor(
    image_path: Optional[str],
    fallback_frame: Optional[Image.Image],
    width: int,
    height: int,
    first_frame_video: Optional[str] = None,
) -> Optional[torch.Tensor]:
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read reference image: {image_path}")
        image = cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    elif first_frame_video:
        tensor = load_video_frame_tensor(first_frame_video, width, height, frame_index=1)
        if tensor is not None:
            return tensor
        if fallback_frame is None:
            return None
        image = np.array(fallback_frame.resize((width, height), Image.BICUBIC))
    else:
        if fallback_frame is None:
            return None
        image = np.array(fallback_frame.resize((width, height), Image.BICUBIC))

    tensor = torch.from_numpy(image.astype(np.uint8)).unsqueeze(0).unsqueeze(0).contiguous()
    return tensor


def pil_to_tensor_batch(frame: Image.Image, width: int, height: int) -> torch.Tensor:
    resized = frame.resize((width, height), Image.BICUBIC)
    arr = np.array(resized).astype(np.uint8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def extend_frames(frames: List[Image.Image], target_length: int) -> List[Image.Image]:
    if not frames:
        return frames
    if len(frames) >= target_length:
        return frames[:target_length]
    extension = [frames[-1].copy() for _ in range(target_length - len(frames))]
    return frames + extension


def resize_to_aspect(frames: Iterable[Image.Image], original_size: Tuple[int, int], target_height: int) -> List[Image.Image]:
    original_height, original_width = original_size
    if original_height <= 0 or original_width <= 0:
        return list(frames)
    target_width = int(round(original_width / original_height * target_height))
    resized = [frame.resize((target_width, target_height), Image.BICUBIC) for frame in frames]
    return resized


def run_single_pass(
    pipe: ControlCogVideoXPipeline,
    source_images: torch.Tensor,
    target_images: Optional[torch.Tensor],
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    frames_per_pass: int,
    steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    device: torch.device,
) -> List[Image.Image]:
    pipe.vae.to(device)
    pipe.transformer.to(device)
    pipe.controlnet_transformer.to(device)

    source_pixel_values = source_images.to(torch.float32) / 127.5 - 1.0
    source_pixel_values = source_pixel_values.to(dtype=torch.float16, device=device)

    if target_images is not None:
        target_pixel_values = target_images.to(torch.float32) / 127.5 - 1.0
        target_pixel_values = target_pixel_values.to(dtype=torch.float16, device=device)
    else:
        target_pixel_values = None

    with torch.no_grad():
        source_latents = pipe.vae.encode(rearrange(source_pixel_values, "b f h w c -> b c f h w")).latent_dist.sample()
        source_latents = source_latents.to(torch.float16) * pipe.vae.config.scaling_factor
        source_latents = rearrange(source_latents, "b c f h w -> b f c h w")

        if target_pixel_values is not None:
            target_latents = pipe.vae.encode(rearrange(target_pixel_values, "b f h w c -> b c f h w")).latent_dist.sample()
            target_latents = target_latents.to(torch.float16) * pipe.vae.config.scaling_factor
            target_latents = rearrange(target_latents, "b c f h w -> b f c h w")
            target_latents = torch.cat([target_latents, torch.zeros_like(source_latents)[:, 1:]], dim=1)
        else:
            target_latents = None

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video_condition=source_latents,
            video_condition2=target_latents,
            height=height,
            width=width,
            num_frames=frames_per_pass,
            num_inference_steps=steps,
            interval=6,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

    return output


def run_sequential_generation(
    pipe: ControlCogVideoXPipeline,
    source_images: torch.Tensor,
    initial_target: Optional[torch.Tensor],
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    total_frames: int,
    frames_per_pass: int,
    steps: int,
    guidance_scale: float,
    base_seed: int,
    device: torch.device,
) -> List[Image.Image]:
    results: List[Image.Image] = []
    target_tensor = initial_target
    passes = max(1, math.ceil(total_frames / frames_per_pass))

    for pass_idx in range(passes):
        remaining = total_frames - len(results)
        if remaining <= 0:
            break

        generator = torch.Generator(device=device).manual_seed(base_seed + pass_idx)
        frames = run_single_pass(
            pipe=pipe,
            source_images=source_images,
            target_images=target_tensor,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            frames_per_pass=frames_per_pass,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            device=device,
        )

        frames = frames[:remaining]
        results.extend(frames)

        if results and pass_idx + 1 < passes:
            target_tensor = pil_to_tensor_batch(results[-1], width, height)

    return results[:total_frames]


def save_video(frames: List[Image.Image], path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    export_to_video(frames, path, fps=fps)


def save_side_by_side(
    input_frames: List[Image.Image],
    output_frames: List[Image.Image],
    path: str,
    fps: int,
) -> None:
    count = min(len(input_frames), len(output_frames))
    combined: List[Image.Image] = []
    for idx in range(count):
        left = input_frames[idx]
        right = output_frames[idx]
        if left.size[1] != right.size[1]:
            right = right.resize((int(right.size[0] * left.size[1] / right.size[1]), left.size[1]), Image.BICUBIC)
        canvas = Image.new("RGB", (left.width + right.width, left.height))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        combined.append(canvas)
    save_video(combined, path, fps)


def derive_base_name(item: Dict[str, Any], index: int) -> str:
    if "item_id" in item and item["item_id"]:
        return str(item["item_id"])
    if "output_basename" in item and item["output_basename"]:
        return str(item["output_basename"])
    if "task_type" in item and "sample_id" in item:
        return f"{item['task_type']}_{item['sample_id']}"
    if "file_name" in item:
        return os.path.splitext(os.path.basename(str(item["file_name"])))[0]
    if "source_video_path" in item:
        return os.path.splitext(os.path.basename(str(item["source_video_path"])))[0]
    if "id" in item:
        return f"sample_{item['id']}"
    return f"sample_{index:04d}"


def load_tasks(tasks_json: str) -> List[Dict[str, Any]]:
    with open(tasks_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            return data["tasks"]
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {tasks_json}")


def gather_items(tasks: List[Dict[str, Any]]) -> List[TaskItem]:
    items: List[TaskItem] = []
    for idx, task in enumerate(tasks):
        base = derive_base_name(task, idx)
        items.append(TaskItem(base_name=base, data=task))
    return items


def main() -> None:
    args = parse_args()
    tasks_json_abs = os.path.abspath(args.tasks_json)
    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    edited_videos_dir = os.path.abspath(args.edited_videos_dir) if args.edited_videos_dir else None

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Running distributed inference on {world_size} GPUs")
        print(f"Tasks JSON: {tasks_json_abs}")
        print(f"Output directory: {output_dir_abs}")
        if edited_videos_dir:
            print(f"Edited videos directory: {edited_videos_dir}")

    tasks = load_tasks(tasks_json_abs)
    items = gather_items(tasks)

    pending: List[TaskItem] = []
    for item in items:
        output_path = os.path.join(output_dir_abs, f"{item.base_name}_gen.mp4")
        if args.skip_existing and os.path.exists(output_path):
            continue
        pending.append(item)

    if rank == 0:
        skipped = len(items) - len(pending)
        print(f"Total tasks: {len(items)} | Existing skipped: {skipped} | Pending: {len(pending)}")

    subset = pending[rank::world_size]
    if rank == 0:
        for r in range(world_size):
            count = len(pending[r::world_size])
            print(f"GPU {r} will process {count} item(s)")

    if not subset:
        if rank == 0:
            print("No pending tasks. Exiting.")
        dist.barrier()
        dist.destroy_process_group()
        return

    pipeline = init_pipeline(
        model_root=os.path.abspath(args.model_root),
        control_checkpoint=os.path.abspath(args.control_checkpoint),
        device=device,
        dtype=torch.float16,
        control_num_layers=args.control_num_layers,
    )

    base_dir = os.path.dirname(tasks_json_abs)

    for local_index, item in enumerate(subset):
        task = item.data
        base_name = item.base_name
        output_video_path = os.path.join(output_dir_abs, f"{base_name}_gen.mp4")
        if args.skip_existing and os.path.exists(output_video_path):
            if rank == 0:
                print(f"[GPU {rank}] Skipping {base_name}, output already exists.")
            continue

        source_video_path = (
            task.get("source_video_path")
            or task.get("video_path")
            or task.get("input_video")
            or task.get("source_path")
        )
        source_video_path = resolve_path(source_video_path, base_dir)
        if source_video_path is None or not os.path.exists(source_video_path):
            print(f"[GPU {rank}] Missing source video for {base_name}, skipping.")
            continue

        reference_image_path = (
            task.get("reference_image_path")
            or task.get("image_path")
            or task.get("target_image_path")
            or task.get("first_frame_path")
        )
        reference_image_path = resolve_path(reference_image_path, base_dir)

        edited_video_path = None
        if edited_videos_dir:
            candidate = os.path.join(edited_videos_dir, f"gen_{base_name}.mp4")
            if os.path.exists(candidate):
                edited_video_path = candidate

        prompt = (
            task.get("positive_prompt")
            or task.get("prompt")
            or task.get("edit_instruction")
            or task.get("text")
            or ""
        )
        negative_prompt = task.get("negative_prompt", args.negative_prompt)
        guidance_scale = float(task.get("guidance_scale", args.guidance_scale))
        steps = int(task.get("num_inference_steps", args.steps))
        total_frames = int(task.get("num_frames", args.num_frames))
        frames_per_pass = int(task.get("source_frames", args.source_frames))
        height = int(task.get("height", args.height))
        width = int(task.get("width", args.width))
        fps = int(task.get("fps", args.fps))
        sample_seed = int(task.get("seed", args.seed + rank * 1000 + local_index))

        print(f"[GPU {rank}] Processing {base_name} | prompt: {prompt[:80]}...")

        try:
            source_tensor, original_frames, resized_frames, original_size = load_video_frames(
                source_video_path,
                target_frames=frames_per_pass,
                width=width,
                height=height,
            )
        except Exception as exc:
            print(f"[GPU {rank}] Failed to load video for {base_name}: {exc}")
            continue

        try:
            reference_tensor = prepare_reference_tensor(
                reference_image_path,
                fallback_frame=original_frames[0] if original_frames else None,
                width=width,
                height=height,
                first_frame_video=edited_video_path,
            )
        except Exception as exc:
            print(f"[GPU {rank}] Failed to load reference image for {base_name}: {exc}")
            continue

        try:
            generated_frames = run_sequential_generation(
                pipe=pipeline,
                source_images=source_tensor,
                initial_target=reference_tensor,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                total_frames=total_frames,
                frames_per_pass=frames_per_pass,
                steps=steps,
                guidance_scale=guidance_scale,
                base_seed=sample_seed,
                device=device,
            )
        except Exception as exc:
            print(f"[GPU {rank}] Generation failed for {base_name}: {exc}")
            continue

        if not generated_frames:
            print(f"[GPU {rank}] No frames generated for {base_name}, skipping save.")
            continue

        input_frames_for_save = extend_frames(resized_frames, len(generated_frames))

        output_frames_resized = resize_to_aspect(generated_frames, original_size, target_height=height)
        input_frames_resized = resize_to_aspect(input_frames_for_save, original_size, target_height=height)

        input_path = os.path.join(output_dir_abs, f"{base_name}_input.mp4")
        compare_path = os.path.join(output_dir_abs, f"{base_name}_compare.mp4")
        info_path = os.path.join(output_dir_abs, f"{base_name}_gen_info.txt")

        try:
            save_video(input_frames_resized, input_path, fps)
            save_video(output_frames_resized, output_video_path, fps)
            save_side_by_side(input_frames_resized, output_frames_resized, compare_path, fps)
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write(prompt)
        except Exception as exc:
            print(f"[GPU {rank}] Saving results failed for {base_name}: {exc}")
            continue

        print(f"[GPU {rank}] Completed {base_name}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass


