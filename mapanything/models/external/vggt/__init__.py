# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for VGGT
"""

import torch

from mapanything.models.external.pi3.layers.frame_selection import FrameSelector
from mapanything.models.external.vggt.models.vggt import VGGT
from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


class VGGTWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        depth=24,
        num_heads=16,
        intermediate_layer_idx=[4, 11, 17, 23],
        load_custom_ckpt=False,
        custom_ckpt_path=None,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path

        if load_pretrained_weights:
            # Load pre-trained weights
            if not torch_hub_force_reload:
                # Initialize the 1B VGGT model from huggingface hub cache
                print("Loading facebook/VGGT-1B from huggingface cache ...")
                self.model = VGGT.from_pretrained(
                    "facebook/VGGT-1B",
                    local_files_only=True,
                )
            else:
                # Initialize the 1B VGGT model
                print("Re-downloading facebook/VGGT-1B ...")
                self.model = VGGT.from_pretrained(
                    "facebook/VGGT-1B", force_download=True
                )
        else:
            # Load the VGGT class
            self.model = VGGT(
                depth=depth,
                num_heads=num_heads,
                intermediate_layer_idx=intermediate_layer_idx,
            )

        # Get the dtype for VGGT inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Load custom checkpoint if requested
        if self.load_custom_ckpt:
            print(f"Loading checkpoint from {self.custom_ckpt_path} ...")
            assert self.custom_ckpt_path is not None, (
                "custom_ckpt_path must be provided if load_custom_ckpt is set to True"
            )
            custom_ckpt = torch.load(self.custom_ckpt_path, weights_only=False)
            print(self.model.load_state_dict(custom_ckpt, strict=True))
            del custom_ckpt  # in case it occupies memory

        # Remove gradient checkpointing from DINOv2 backbone blocks.
        # The aggregator wraps every patch_embed block with checkpoint() at
        # construction time, which is useful for training but adds overhead
        # during inference even under torch.no_grad().
        patch_embed = self.model.aggregator.patch_embed
        if hasattr(patch_embed, "blocks"):
            for block in patch_embed.blocks:
                if hasattr(block.__class__, "_restore_cls"):
                    block.__class__ = block.__class__._restore_cls

    def forward(self, views, frame_selector=None, token_downsample=1, token_keep_rate=1.0, token_selection_method="diverse", ds_keep_register=False, backbone_minibatch_size=0, global_as_frame_layers=None, global_as_meanpool_layers=None, layer_config=None, adaptive_ds_threshold=None, adaptive_frame_threshold=None):
        """
        Forward pass wrapper for VGGT

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]
            frame_selector: Optional frame selection config (dict or FrameSelector).
                            If None, uses full cross-frame attention (original behavior).
            token_downsample: Spatial stride factor for K/V token downsampling.
                Can be used independently of frame_selector. Default: 1 (no downsampling).

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Convert dict to FrameSelector if needed
        if isinstance(frame_selector, dict):
            frame_selector = FrameSelector(**frame_selector)
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # VGGT expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "VGGT expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Run the VGGT aggregator
        # Only retain the intermediate outputs needed by the heads to save ~2GB memory.
        # depth_head uses intermediate_layer_idx, camera_head uses [-1] (= depth-1).
        retain_indices = set(self.model.depth_head.intermediate_layer_idx)
        retain_indices.add(self.model.aggregator.depth - 1)

        with torch.autocast("cuda", dtype=self.dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(
                images, retain_indices=retain_indices, frame_selector=frame_selector,
                token_downsample=token_downsample, token_keep_rate=token_keep_rate,
                token_selection_method=token_selection_method,
                ds_keep_register=ds_keep_register,
                backbone_minibatch_size=backbone_minibatch_size,
                global_as_frame_layers=global_as_frame_layers,
                global_as_meanpool_layers=global_as_meanpool_layers,
                layer_config=layer_config,
                adaptive_ds_threshold=adaptive_ds_threshold,
                adaptive_frame_threshold=adaptive_frame_threshold,
            )

        # Run the Camera + Pose Branch of VGGT
        with torch.autocast("cuda", enabled=False):
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # Extrinsics Shape: (B, V, 3, 4)
            # Intrinsics Shape: (B, V, 3, 3)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )

            # Predict Depth Maps
            # Depth Shape: (B, V, H, W, 1)
            # Depth Confidence Shape: (B, V, H, W)
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images, ps_idx
            )

            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics, intrinsics, depth map for the current view
                curr_view_extrinsic = extrinsic[:, view_idx, ...]
                curr_view_extrinsic = closed_form_inverse_se3(
                    curr_view_extrinsic
                )  # Convert to cam2world
                curr_view_intrinsic = intrinsic[:, view_idx, ...]
                curr_view_depth_z = depth_map[:, view_idx, ...]
                curr_view_depth_z = curr_view_depth_z.squeeze(-1)
                curr_view_confidence = depth_conf[:, view_idx, ...]

                # Get the camera frame pointmaps
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the extrinsics to quaternions and translations
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

                # Get the pointmaps
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list
                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                    }
                )

        return res
