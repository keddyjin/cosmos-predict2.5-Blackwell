# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action-conditioned Video2World inference script."""

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedInferenceOverrides,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import (
    handle_tyro_exception,
    is_rank0,
)

import torch
from einops import rearrange
import vfly
from vfly.layers import VflyAttnProcessor, apply_vfly_linear, apply_vfly_norm


class VflyCosmosAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = VflyAttnProcessor().vfly_attn

    def set_context_parallel_group(self, process_group, ranks, stream):
        # cp will be automatically handled by vfly
        # this function is a placeholder to satisfy the interface
        pass

    def forward(self, query, key, value, *args, **kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0

        # Follow Cosmos's implementation to use bfloat16
        dtype=torch.bfloat16
        query = query.to(dtype)
        key = key.to(dtype)
        value = value.to(dtype)

        results = self.attn(query, key, value, tensor_layout="NHD")
        return rearrange(results, "b ... h l -> b ... (h l)")

def enable_vfly(pipe, args, cp_method="ulysses"):
    cp_size = args.context_parallel_size
    # VFly supported 3 cp methods: cp(allgather kv), ring, ulysses
    if cp_method == "cp":
        cp_key = "dit_cp_size"
    elif cp_method == "ring":
        cp_key = "dit_ring_size"
    elif cp_method == "ulysses":
        cp_key = "dit_ulysses_size"
    else:
        raise ValueError(f"Invalid cp method: {cp_method}")
    vfly.setup_configs(
        parallel = {
            cp_key: cp_size,
        },
        attn = {
            "type": "sage-attn",
        }
    )
    # replace the attention with vfly_cosmos_attention
    for module in pipe.model.modules():
        if hasattr(module, "attn_op"):
            setattr(module, "attn_op", VflyCosmosAttention())

    apply_vfly_linear(pipe.model, load_parameters=True)
    apply_vfly_norm(pipe.model, rmsnorm=["q_norm", "k_norm"], load_parameters=True)

    with open("model_action_conditioned_vfly.txt", "w") as f:
        f.write(str(pipe.model))


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_file: Annotated[Path, tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file"""
    setup: ActionConditionedSetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: ActionConditionedInferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""

from cosmos_predict2.action_conditioned import *

def vfly_inference(
    setup_args: ActionConditionedSetupArguments,
    inference_args: ActionConditionedInferenceArguments,):
    """Run action-conditioned video generation inference using resolved setup and per-run arguments."""
    torch.enable_grad(False)  # Disable gradient calculations for inference

    # Validate num_latent_conditional_frames at the very beginning
    if inference_args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, but got {inference_args.num_latent_conditional_frames}"
        )

    # Determine supported extensions based on num_latent_conditional_frames
    if inference_args.num_latent_conditional_frames > 1:
        # Check if input folder contains any videos
        has_videos = False
        for file_name in os.listdir(inference_args.input_root):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in VIDEO_EXTENSIONS:
                has_videos = True
                break

        if not has_videos:
            raise ValueError(
                f"num_latent_conditional_frames={inference_args.num_latent_conditional_frames} > 1 requires video inputs, "
                f"but no videos found in {inference_args.input_root}. Found extensions: "
                f"{set(os.path.splitext(f)[1].lower() for f in os.listdir(inference_args.input_root) if os.path.splitext(f)[1])}"
            )

        logger.info(f"Using video-only mode with {inference_args.num_latent_conditional_frames} conditional frames")
    elif inference_args.num_latent_conditional_frames == 1:
        logger.info(f"Using image+video mode with {inference_args.num_latent_conditional_frames} conditional frame")

    # Get checkpoint and experiment from setup args
    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    # pyrefly: ignore  # missing-attribute
    checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

    # Ensure experiment is not None
    if experiment is None:
        raise ValueError("Experiment name must be provided either in setup args or checkpoint metadata")

    # Initialize the inference handler with context parallel support
    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        # pyrefly: ignore  # bad-argument-type
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Load action loading function
    action_load_fn = load_callable(inference_args.action_load_fn)

    # Get input video and annotation paths
    input_video_path = inference_args.input_root
    input_json_path = inference_args.input_root / inference_args.input_json_sub_folder
    input_json_list = glob(str(input_json_path / "*.json"))

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Ensure save directory exists
    inference_args.save_root.mkdir(parents=True, exist_ok=True)

    # Process each file in the input directory
    for annotation_path in input_json_list[inference_args.start : inference_args.end]:
        with open(annotation_path, "r") as f:
            json_data = json.load(f)

        # Convert camera_id to integer if it's a string and can be converted to an integer
        camera_id = (
            int(inference_args.camera_id)
            if isinstance(inference_args.camera_id, str) and inference_args.camera_id.isdigit()
            else inference_args.camera_id
        )

        if isinstance(json_data["videos"][camera_id], dict):
            video_path = str(input_video_path / json_data["videos"][camera_id]["video_path"])
        else:
            video_path = str(input_video_path / json_data["videos"][camera_id])

        # Load action data using the configured function
        action_data = action_load_fn()(json_data, video_path, inference_args)
        actions = action_data["actions"]
        img_array = action_data["initial_frame"]

        img_name = annotation_path.split("/")[-1].split(".")[0]

        frames = [img_array]
        chunk_video = []

        video_name = str(inference_args.save_root / f"{img_name.replace('.jpg', '.mp4')}")
        chunk_video_name = str(inference_args.save_root / f"{img_name}_chunk.mp4")
        logger.info(f"Saving video to {video_name}")
        if os.path.exists(chunk_video_name):
            logger.info(f"Video already exists: {chunk_video_name}")
            continue

        for i in range(inference_args.start_frame_idx, len(actions), inference_args.chunk_size):
            # Handle incomplete chunks
            actions_chunk = actions[i : i + inference_args.chunk_size]
            if actions_chunk.shape[0] != inference_args.chunk_size:
                pad_len = inference_args.chunk_size - actions_chunk.shape[0]
                if pad_len > 0:
                    action_shape = list(actions.shape[1:])
                    pad_shape = [pad_len] + action_shape
                    pad_actions = np.zeros(pad_shape, dtype=actions.dtype)
                    actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

            # Convert img_array to tensor and prepare video input
            # pyrefly: ignore  # implicit-import
            img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)
            num_video_frames = actions_chunk.shape[0] + 1
            vid_input = torch.cat(
                [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
            )
            vid_input = (vid_input * 255.0).to(torch.uint8)
            vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

            # Call generate_vid2world
            video = video2world_cli.generate_vid2world(
                prompt=inference_args.prompt or "",
                input_path=vid_input,
                action=torch.from_numpy(actions_chunk).float()
                if isinstance(actions_chunk, np.ndarray)
                else actions_chunk,
                guidance=inference_args.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=inference_args.num_latent_conditional_frames,
                resolution=inference_args.resolution,
                seed=i,
                negative_prompt=inference_args.negative_prompt,
            )
            # Extract next frame and video from result
            video_normalized = (video - (-1)) / (1 - (-1))
            video_clamped = (
                (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            )
            next_img_array = video_clamped[-1]  # Last frame is the next frame
            frames.append(next_img_array)
            img_array = next_img_array
            chunk_video.append(video_clamped)

            if inference_args.single_chunk:
                break

        chunk_list = [chunk_video[0]] + [
            chunk_video[i][: inference_args.chunk_size] for i in range(1, len(chunk_video))
        ]
        chunk_video = np.concatenate(chunk_list, axis=0)
        if inference_args.single_chunk:
            chunk_video_name = str(inference_args.save_root / f"{img_name}_single_chunk.mp4")
        else:
            chunk_video_name = str(inference_args.save_root / f"{img_name}_chunk.mp4")

        if rank0:
            mediapy.write_video(chunk_video_name, chunk_video, fps=inference_args.save_fps)
            logger.info(f"Saved video to {chunk_video_name}")

    # Synchronize all processes before cleanup
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    video2world_cli.cleanup()

def main(args: Args) -> None:
    inference_args = ActionConditionedInferenceArguments.from_files([args.input_file], overrides=args.overrides)[0]
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    # from cosmos_predict2.action_conditioned import inference

    # inference(args.setup, inference_args)
    vfly_inference(args.setup, inference_args)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
