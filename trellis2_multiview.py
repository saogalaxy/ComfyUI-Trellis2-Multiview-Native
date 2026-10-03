"""Trellis2 multi-view conditioning (native-style, no compiled deps).

Averages DINOv3 global tokens across 1-4 views and emits the exact same
CONDITIONING format as core `Trellis2Conditioning`, so all downstream core
nodes (Trellis2ShapeStage / KSampler / Trellis2UpsampleStage /
Trellis2TextureStage / VAE decodes) work unchanged.

Why averaging: Trellis2 diffusion weights use global cross-attention
(`image_attn_mode="global"`), trained single-view. There is no projection /
camera-fusion path for Trellis2 (that is the Pixal3D route with NAF +
transform matrices). Mean-pooling the global tokens is the architecturally
consistent way to condition Trellis2 on several views without retraining.
View order is irrelevant to the diffusion; the first connected view is
treated as "front" only as a posing convention for the mesh output.
"""

from comfy_api.latest import ComfyExtension, IO, Types
import comfy.model_management
import comfy.utils
import logging
import torch
from typing_extensions import override

_VIEW_ORDER = ("front", "left", "back", "right")


def _dinov3_encode_global(model, image_bchw, image_size):
    model_internal = model.model
    device = comfy.model_management.get_torch_device()
    img_t = comfy.utils.common_upscale(
        image_bchw, image_size, image_size, "lanczos", "disabled"
    ).to(device)
    mean = torch.tensor(
        model.image_mean or [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        model.image_std or [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    img_t = (img_t - mean) / std
    return model_internal(img_t, skip_norm_elementwise=True)[0]


def _to_bchw(frame_hwc):
    if frame_hwc.shape[-1] == 4:
        rgb = frame_hwc[..., :3] * frame_hwc[..., 3:4]
    else:
        rgb = frame_hwc[..., :3]
    return rgb.movedim(-1, -3).unsqueeze(0).contiguous().float().clamp(0, 1)


class Trellis2MultiViewConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        views = [
            IO.Image.Input(
                name,
                optional=True,
                tooltip=(
                    f"Square {name} view. First ACTIVE view "
                    "(front, left, back, right order) sets the mesh front. "
                    "Views are appearance-averaged; geometry fusion is not "
                    "performed (Trellis2 is global-attention only)."
                ),
            )
            for name in _VIEW_ORDER
        ]
        switches = [
            IO.Boolean.Input(
                f"use_{name}",
                default=True,
                tooltip=(
                    f"Include the {name} view in the average. "
                    "Turn off to skip a wired view without disconnecting it."
                ),
            )
            for name in _VIEW_ORDER
        ]
        return IO.Schema(
            node_id="Trellis2MultiViewConditioning",
            display_name="Trellis2 Multi-View Conditioning",
            category="model/conditioning/trellis",
            inputs=[
                IO.ClipVision.Input(
                    "clip_vision_model",
                    tooltip="DINOv3 ViT-L/16 ClipVision (same as core Trellis2).",
                ),
            ]
            + views
            + switches,
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(
        cls, clip_vision_model, front=None, left=None, back=None, right=None,
        use_front=True, use_left=True, use_back=True, use_right=True,
    ) -> IO.NodeOutput:
        views = {"front": front, "left": left, "back": back, "right": right}
        enabled = {"front": use_front, "left": use_left,
                   "back": use_back, "right": use_right}
        names = [n for n in _VIEW_ORDER
                 if views[n] is not None and enabled.get(n, True)]
        skipped = [n for n in _VIEW_ORDER
                   if views[n] is not None and not enabled.get(n, True)]
        if skipped:
            logging.info(
                "Trellis2MultiViewConditioning: view switch(es) off, "
                "skipping: %s", ", ".join(skipped))
        if not names:
            raise ValueError(
                "Trellis2MultiViewConditioning needs at least one "
                "connected view with its use_* switch on"
            )
        if len(names) == 1:
            logging.warning(
                "Trellis2MultiViewConditioning: single view connected, "
                "output equals core Trellis2Conditioning."
            )
        if names[0] != "front":
            logging.warning(
                "Trellis2MultiViewConditioning: no front view, mesh will be "
                f"posed with the {names[0]} view as its front."
            )

        batch_size = views[names[0]].shape[0]
        num_views = len(names)
        out_device = comfy.model_management.intermediate_device()
        comfy.model_management.load_model_gpu(clip_vision_model.patcher)

        avg_512, avg_1024 = [], []
        for b in range(batch_size):
            t512, t1024 = [], []
            for name in names:
                view = views[name]
                frame = view[b % view.shape[0]]
                bchw = _to_bchw(frame)
                t512.append(_dinov3_encode_global(clip_vision_model, bchw, 512))
                t1024.append(
                    _dinov3_encode_global(clip_vision_model, bchw, 1024)
                )
            avg_512.append(torch.stack(t512, dim=0).mean(dim=0).to(out_device))
            avg_1024.append(
                torch.stack(t1024, dim=0).mean(dim=0).to(out_device)
            )

        cond_512 = torch.cat(avg_512, dim=0)
        cond_1024 = torch.cat(avg_1024, dim=0)
        positive = [[cond_512, {"embeds": cond_1024}]]
        negative = [
            [
                torch.zeros_like(cond_512),
                {"embeds": torch.zeros_like(cond_1024)},
            ]
        ]
        logging.info(
            "Trellis2MultiViewConditioning: averaged %d view(s) over %d "
            "item(s); cond %s / embeds %s.",
            num_views,
            batch_size,
            tuple(cond_512.shape),
            tuple(cond_1024.shape),
        )
        return IO.NodeOutput(positive, negative)


class MeshWithVoxelToNativeBridge(IO.ComfyNode):
    """Bridge visualbruno Trellis2 MESHWITHVOXEL to native ComfyUI MESH + VOXEL.

    Replaces the fragile file round-trip (Trellis2ExportMesh -> GLB on disk ->
    Trellis2MeshEncoder re-encode with shape VAE + encoder weights) with a
    direct in-memory conversion:

    - MESH: vertices/faces -> core Types.MESH batch (1, N, 3) / (1, M, 3).
    - VOXEL: grid coords/attrs -> core Types.VOXEL (data [N, 4] with batch
      column, voxel_colors [N, C], resolution R), ready for BakeTextureFromVoxel.

    Visualbruno Trellis2 outputs Z-up; native ComfyUI MESH/VOXEL, WTiVo and
    Bake expect Y-up. With reorient_zup_to_yup=True (default) this applies the
    same R_x(-90) as Trellis2MeshWithVoxelToTrimesh '90 degrees':
    vertices (x, y, z) -> (x, z, -y) and voxel indices (x, y, z) ->
    (x, z, R-1-y).
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshWithVoxelToNativeBridge",
            display_name="Trellis2 MeshWithVoxel to Native Bridge",
            category="3d/mesh/bridge",
            description=(
                "Converts visualbruno Trellis2 MESHWITHVOXEL to native "
                "ComfyUI MESH + VOXEL in memory. Feed MESH into "
                "MeshReconstructWithQuad / WTiVo / LODTailor and VOXEL into "
                "BakeTextureFromVoxel. No GLB export, no shape-encoder "
                "re-encode, no VAE needed."
            ),
            inputs=[
                IO.Custom("MESHWITHVOXEL").Input(
                    "mesh",
                    tooltip="MeshWithVoxel from Trellis2 generators / remesh / fill-holes.",
                ),
                IO.Boolean.Input(
                    "reorient_zup_to_yup",
                    default=True,
                    tooltip=(
                        "Apply Z-up to Y-up (matches ToTrimesh '90 degrees'). "
                        "Keep on for WTiVo / Bake / SaveGLB."
                    ),
                ),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Voxel.Output(display_name="voxel_colors"),
            ],
        )

    @classmethod
    def execute(cls, mesh, reorient_zup_to_yup=True) -> IO.NodeOutput:
        verts = mesh.vertices.detach().float().cpu()
        faces = mesh.faces.detach().int().cpu()
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                "MeshWithVoxelToNativeBridge: expected vertices [N, 3], "
                f"got {tuple(verts.shape)}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                "MeshWithVoxelToNativeBridge: expected faces [M, 3], "
                f"got {tuple(faces.shape)}"
            )

        coords = getattr(mesh, "coords", None)
        attrs = getattr(mesh, "attrs", None)
        if coords is None or attrs is None:
            logging.warning(
                "MeshWithVoxelToNativeBridge: input has no voxel grid "
                "(coords/attrs are None) - generator ran with "
                "'generate_texture_slat' disabled. Passing MESH through and "
                "emitting an EMPTY voxel grid; BakeTextureFromVoxel will bake "
                "black maps. Enable 'generate_texture_slat' on the Trellis2 "
                "generator for real textures."
            )
            voxel_size = getattr(mesh, "voxel_size", None)
            resolution = 1024
            try:
                if isinstance(voxel_size, (float, int)) and float(voxel_size) > 0:
                    resolution = int(round(1.0 / float(voxel_size)))
            except Exception:
                resolution = 1024
            if bool(reorient_zup_to_yup):
                verts = torch.stack(
                    [verts[:, 0], verts[:, 2], -verts[:, 1]], dim=-1
                ).contiguous()
            native_mesh = Types.MESH(
                vertices=verts.unsqueeze(0).contiguous(),
                faces=faces.unsqueeze(0).contiguous(),
            )
            native_voxel = Types.VOXEL(
                torch.zeros((0, 4), dtype=torch.long),
                torch.zeros((0, 6), dtype=torch.float32),
                resolution,
            )
            return IO.NodeOutput(native_mesh, native_voxel)
        coords = coords.detach().cpu()
        attrs = attrs.detach().float().cpu()
        if coords.shape[0] != attrs.shape[0]:
            raise ValueError(
                "MeshWithVoxelToNativeBridge: coords/attrs row mismatch: "
                f"{tuple(coords.shape)} vs {tuple(attrs.shape)}"
            )
        if coords.shape[1] == 4:
            coords = coords[:, 1:]
        if coords.shape[1] != 3:
            raise ValueError(
                "MeshWithVoxelToNativeBridge: expected grid coords [N, 3], "
                f"got {tuple(coords.shape)}"
            )

        voxel_size = getattr(mesh, "voxel_size", None)
        resolution = None
        try:
            if isinstance(voxel_size, (float, int)):
                resolution = int(round(1.0 / float(voxel_size)))
            elif isinstance(voxel_size, torch.Tensor):
                resolution = int(round(1.0 / float(voxel_size.flatten()[0].item())))
        except Exception:
            resolution = None
        if resolution is None or resolution <= 0:
            cmax = int(coords.long().max().item()) + 1
            resolution = next(
                (r for r in (256, 512, 1024, 1536, 2048) if r >= cmax), cmax
            )

        coords_long = coords.long()
        if bool(reorient_zup_to_yup):
            verts = torch.stack(
                [verts[:, 0], verts[:, 2], -verts[:, 1]], dim=-1
            ).contiguous()
            coords_long = torch.stack(
                [
                    coords_long[:, 0],
                    coords_long[:, 2],
                    (resolution - 1) - coords_long[:, 1],
                ],
                dim=-1,
            ).contiguous()

        native_mesh = Types.MESH(
            vertices=verts.unsqueeze(0).contiguous(),
            faces=faces.unsqueeze(0).contiguous(),
        )
        batch_col = torch.zeros(
            (coords_long.shape[0], 1), dtype=torch.long
        )
        voxel_data = torch.cat([batch_col, coords_long], dim=1).contiguous()
        native_voxel = Types.VOXEL(
            voxel_data, attrs.contiguous(), resolution
        )
        logging.info(
            "MeshWithVoxelToNativeBridge: mesh v%s f%s -> MESH %s / %s; "
            "voxel N=%d C=%d R=%d (reorient=%s).",
            tuple(mesh.vertices.shape),
            tuple(mesh.faces.shape),
            tuple(native_mesh.vertices.shape),
            tuple(native_mesh.faces.shape),
            voxel_data.shape[0],
            attrs.shape[1],
            resolution,
            bool(reorient_zup_to_yup),
        )
        return IO.NodeOutput(native_mesh, native_voxel)


class Trellis2MultiviewExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Trellis2MultiViewConditioning, MeshWithVoxelToNativeBridge]


async def comfy_entrypoint() -> Trellis2MultiviewExtension:
    return Trellis2MultiviewExtension()
