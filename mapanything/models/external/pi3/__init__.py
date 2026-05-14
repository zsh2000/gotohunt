# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Pi3
"""

import torch

from mapanything.models.external.pi3.layers.frame_selection import FrameSelector
from mapanything.models.external.pi3.models.pi3 import Pi3
from mapanything.models.external.vggt.utils.rotation import mat_to_quat


class Pi3Wrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        pos_type="rope100",
        decoder_size="large",
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload

        if load_pretrained_weights:
            # Load pre-trained weights
            if not torch_hub_force_reload:
                # Initialize the Pi3 model from huggingface hub cache
                print("Loading Pi3 from huggingface cache ...")
                self.model = Pi3.from_pretrained(
                    "yyfz233/Pi3",
                )
            else:
                # Initialize the Pi3 model
                self.model = Pi3.from_pretrained("yyfz233/Pi3", force_download=True)
        else:
            # Load the Pi3 class
            self.model = Pi3(
                pos_type=pos_type,
                decoder_size=decoder_size,
            )

        # Get the dtype for Pi3 inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    def forward(self, views, frame_selector=None, token_downsample=1, token_keep_rate=1.0, token_selection_method="diverse", ds_keep_register=False, backbone_minibatch_size=0, layer_config=None, adaptive_ds_threshold=None, adaptive_frame_threshold=None, tau=1.0, tau_layers=None):
        """
        Forward pass wrapper for Pi3

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]
            frame_selector: Optional FrameSelector instance for sparse cross-view attention.
                Can also be a dict with keys matching FrameSelector.__init__ args, e.g.:
                    {"strategy": "topk", "top_k": 10}
                    {"strategy": "even", "top_k": 10, "include_self": True}
                    {"strategy": "topk", "top_k": 10, "use_covisibility": True,
                     "covisibility_matrix": matrix, "covisibility_percentile": 50.0}
                If None, uses full attention (original behavior).
            token_downsample: Spatial stride factor for K/V token downsampling.
                Can be used independently of frame_selector. Default: 1 (no downsampling).

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # Pi3 expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "Pi3 expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Create FrameSelector from dict if needed
        if isinstance(frame_selector, dict):
            frame_selector = FrameSelector(**frame_selector)

        # Run the Pi3 aggregator
        with torch.autocast("cuda", dtype=self.dtype):
            results = self.model(images, frame_selector=frame_selector, token_downsample=token_downsample, token_keep_rate=token_keep_rate, token_selection_method=token_selection_method, ds_keep_register=ds_keep_register, backbone_minibatch_size=backbone_minibatch_size, layer_config=layer_config, adaptive_ds_threshold=adaptive_ds_threshold, adaptive_frame_threshold=adaptive_frame_threshold, tau=tau, tau_layers=tau_layers)

        # Need high precision for transformations
        with torch.autocast("cuda", enabled=False):
            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics
                curr_view_extrinsic = results["camera_poses"][:, view_idx, ...]
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Get the depth along ray, ray directions, local point cloud & global point cloud
                curr_view_pts3d_cam = results["local_points"][:, view_idx, ...]
                curr_view_depth_along_ray = torch.norm(
                    curr_view_pts3d_cam, dim=-1, keepdim=True
                )
                curr_view_ray_dirs = curr_view_pts3d_cam / curr_view_depth_along_ray
                curr_view_pts3d = results["points"][:, view_idx, ...]

                # Get the confidence
                curr_view_confidence = results["conf"][:, view_idx, ...]

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
