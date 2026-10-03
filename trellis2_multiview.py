"""Trellis2 multi-view conditioning (native-style, no compiled deps).

Nodes:
- `Trellis2MultiViewConditioning`: averages DINOv3 global tokens across
  1-4 views; same CONDITIONING format as core `Trellis2Conditioning`.
- `Trellis2SpatialMultiViewPatch`: native port of visualbruno's multiview
  sampler fusion. Patches MODEL so every KSampler step runs once per view
  and blends with spatial softmax weights (`front_axis`,
  `blend_temperature`). This is what locks the mesh to the source pose;
  averaging alone cannot. Pure torch, no compiled deps.
- `MeshWithVoxelToNativeBridge` / `NativeMeshVoxelToMeshWithVoxel`:
  in-memory MESHWITHVOXEL <-> MESH+VOXEL bridges.
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


class MeshWithVoxelShim:
    """Dependency-free MESHWITHVOXEL stand-in (Z-up, visualbruno convention).

    Only carries what MeshWithVoxelToNativeBridge reads:
    vertices [N,3], faces [M,3], coords [K,4] (batch col 0),
    attrs [K,C], voxel_size (1/R), layout (None).
    """

    def __init__(self, vertices, faces, coords, attrs, voxel_size):
        self.vertices = vertices
        self.faces = faces
        self.coords = coords
        self.attrs = attrs
        self.voxel_size = voxel_size
        self.layout = None
        self.device = vertices.device if hasattr(vertices, "device") else torch.device("cpu")


class NativeMeshVoxelToMeshWithVoxel(IO.ComfyNode):
    """Pack native MESH + VOXEL (Y-up) into a Z-up MESHWITHVOXEL shim.

    This is the native replacement for visualbruno's
    'Trellis2 - Mesh With Voxel Multi-View Generator' output: it produces
    the same MESHWITHVOXEL type that BRIDGE (MeshWithVoxelToNativeBridge)
    accepts, but from core VaeDecodeShapeTrellis (MESH) +
    VaeDecodeTextureTrellis (VOXEL) with no cumesh / o_voxel / triton.

    Chain: native VAE decodes -> this node -> BRIDGE(reorient=True) ->
    MeshReconstructWithQuad / WTiVo / BakeTextureFromVoxel.
    Y-up -> Z-up here is the exact inverse of the bridge's Z-up -> Y-up:
    verts (x, y, z) -> (x, -z, y), voxels (x, y, z) -> (x, R-1-z, y).
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NativeMeshVoxelToMeshWithVoxel",
            display_name="Trellis2 Native MESH+VOXEL to MeshWithVoxel",
            category="3d/mesh/bridge",
            description=(
                "Packs core native MESH + VOXEL into a Z-up MESHWITHVOXEL "
                "shim that plugs into MeshWithVoxelToNativeBridge. "
                "Native replacement for the visualbruno generator output, "
                "no compiled deps."
            ),
            inputs=[
                IO.Mesh.Input(
                    "mesh",
                    tooltip="MESH from VaeDecodeShapeTrellis (Y-up, batch 1).",
                ),
                IO.Voxel.Input(
                    "voxel_colors",
                    tooltip="VOXEL from VaeDecodeTextureTrellis (Y-up).",
                ),
            ],
            outputs=[
                IO.Custom("MESHWITHVOXEL").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(cls, mesh, voxel_colors) -> IO.NodeOutput:
        verts = mesh.vertices.detach().float().cpu()
        faces = mesh.faces.detach().int().cpu()
        if verts.ndim == 3:
            if verts.shape[0] != 1:
                raise ValueError(
                    "NativeMeshVoxelToMeshWithVoxel: expected batch 1 MESH, "
                    f"got {tuple(verts.shape)}"
                )
            verts = verts[0]
            faces = faces[0] if faces.ndim == 3 else faces
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                "NativeMeshVoxelToMeshWithVoxel: expected vertices [N, 3], "
                f"got {tuple(verts.shape)}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                "NativeMeshVoxelToMeshWithVoxel: expected faces [M, 3], "
                f"got {tuple(faces.shape)}"
            )

        data = voxel_colors.data.detach().cpu()
        feats = voxel_colors.feats.detach().float().cpu()
        resolution = int(voxel_colors.resolution)
        if data.numel() == 0:
            raise ValueError(
                "NativeMeshVoxelToMeshWithVoxel: empty VOXEL (generator ran "
                "without texture?). Enable texture branch before packing."
            )
        if data.shape[1] == 4:
            coords_yup = data[:, 1:].long()
        else:
            coords_yup = data.long()
        if coords_yup.shape[1] != 3:
            raise ValueError(
                "NativeMeshVoxelToMeshWithVoxel: expected voxel coords [K, 3], "
                f"got {tuple(coords_yup.shape)}"
            )
        if coords_yup.shape[0] != feats.shape[0]:
            raise ValueError(
                "NativeMeshVoxelToMeshWithVoxel: coords/attrs row mismatch: "
                f"{tuple(coords_yup.shape)} vs {tuple(feats.shape)}"
            )

        # Y-up -> Z-up (inverse of bridge R_x(-90)).
        verts_zup = torch.stack(
            [verts[:, 0], -verts[:, 2], verts[:, 1]], dim=-1
        ).contiguous()
        coords_zup = torch.stack(
            [
                coords_yup[:, 0],
                (resolution - 1) - coords_yup[:, 2],
                coords_yup[:, 1],
            ],
            dim=-1,
        ).contiguous()
        batch_col = torch.zeros(
            (coords_zup.shape[0], 1), dtype=torch.long
        )
        coords_4 = torch.cat([batch_col, coords_zup], dim=1).contiguous()
        shim = MeshWithVoxelShim(
            vertices=verts_zup,
            faces=faces,
            coords=coords_4,
            attrs=feats.contiguous(),
            voxel_size=1.0 / float(resolution),
        )
        logging.info(
            "NativeMeshVoxelToMeshWithVoxel: MESH v%s f%s + VOXEL N=%d C=%d "
            "R=%d -> MESHWITHVOXEL shim (Z-up).",
            tuple(verts.shape),
            tuple(faces.shape),
            coords_4.shape[0],
            feats.shape[1],
            resolution,
        )
        return IO.NodeOutput(shim)


def _mv_view_scores_sparse(coords_4, resolution, views, front_axis):
    """Per-voxel view scores, ported from visualbruno FlowEulerMultiViewSampler.

    coords_4 [N, 4] layout is [batch, z, y, x] (same argwhere layout both
    stacks use), so col 1 is z and col 3 is x. Pure torch, no deps.
    """
    z = (coords_4[:, 1].float() / float(resolution)) * 2.0 - 1.0
    x = (coords_4[:, 3].float() / float(resolution)) * 2.0 - 1.0
    zero = torch.zeros_like(z)
    if front_axis == "x":
        table = {"front": (x, zero), "back": (-x, zero),
                 "right": (zero, z), "left": (zero, -z)}
    else:
        table = {"front": (zero, z), "back": (zero, -z),
                 "right": (x, zero), "left": (-x, zero)}
    scores = []
    for view in views:
        if view in table:
            a, b = table[view]
            scores.append(a + b)
        else:
            scores.append(torch.full_like(z, -10.0))
    return torch.stack(scores, dim=1)


def _mv_view_scores_dense(shape, device, views, front_axis):
    """Per-voxel view scores for dense [B, C, D, H, W] structure latents."""
    D, H, W = int(shape[2]), int(shape[3]), int(shape[4])
    dz = torch.linspace(-1.0, 1.0, D, device=device)
    dx = torch.linspace(-1.0, 1.0, W, device=device)
    grid_z, _, grid_x = torch.meshgrid(dz, torch.zeros(1, device=device),
                                       dx, indexing="ij")
    grid_z = grid_z.expand(D, H, W)
    grid_x = grid_x.expand(D, H, W)
    if front_axis == "x":
        table = {"front": grid_x, "back": -grid_x,
                 "right": grid_z, "left": -grid_z}
    else:
        table = {"front": grid_z, "back": -grid_z,
                 "right": grid_x, "left": -grid_x}
    scores = []
    for view in views:
        if view in table:
            scores.append(table[view])
        else:
            scores.append(torch.full_like(grid_z, -10.0))
    return torch.stack(scores, dim=0)


class Trellis2SpatialMultiViewPatch(IO.ComfyNode):
    """Native port of visualbruno's multiview sampler fusion (no extra deps).

    Visualbruno's `Trellis2MeshWithVoxelMultiViewGenerator` does NOT average
    views: per sampling step it runs the denoiser once per view and blends
    the velocities with spatial softmax weights
    (`front_axis` + `blend_temperature`, +Z front / +X right when
    `front_axis="z"`). That voxel-space blend is what locks the mesh to the
    source pose. Mean-pooling tokens (the conditioning node) cannot do that.

    This node encodes each view's DINOv3 tokens itself and patches MODEL via
    the supported `set_model_unet_function_wrapper` hook, so every KSampler
    step (structure dense 16^3, shape/texture sparse) runs once per view and
    blends with the same weights. All math is pure torch.

    Chain: UNETLoader -> this node -> KSampler (x4) -> stages/decodes.
    Keep `Trellis2MultiViewConditioning` upstream (its averaged tokens are the
    fallback base cond). Single view connected = passthrough.
    """

    @classmethod
    def define_schema(cls):
        views = [
            IO.Image.Input(
                name,
                optional=True,
                tooltip=f"Square {name} view for spatial fusion.",
            )
            for name in _VIEW_ORDER
        ]
        switches = [
            IO.Boolean.Input(f"use_{name}", default=True)
            for name in _VIEW_ORDER
        ]
        return IO.Schema(
            node_id="Trellis2SpatialMultiViewPatch",
            display_name="Trellis2 Spatial Multi-View Patch",
            category="model/conditioning/trellis",
            inputs=[
                IO.Model.Input("model"),
                IO.ClipVision.Input("clip_vision_model"),
            ]
            + views
            + switches
            + [
                IO.Combo.Input(
                    "front_axis",
                    options=["z", "x"],
                    default="z",
                    tooltip="Matches visualbruno front_axis: which voxel axis the front view looks down.",
                ),
                IO.Float.Input(
                    "blend_temperature",
                    default=2.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    tooltip="Matches visualbruno blend_temperature: higher = harder per-view regions.",
                ),
            ],
            outputs=[IO.Model.Output()],
        )

    @classmethod
    def execute(
        cls, model, clip_vision_model, front=None, left=None, back=None,
        right=None, use_front=True, use_left=True, use_back=True,
        use_right=True, front_axis="z", blend_temperature=2.0,
    ):
        views = {"front": front, "left": left, "back": back, "right": right}
        enabled = {"front": use_front, "left": use_left,
                   "back": use_back, "right": use_right}
        names = [n for n in _VIEW_ORDER
                 if views[n] is not None and enabled.get(n, True)]
        if not names:
            raise ValueError(
                "Trellis2SpatialMultiViewPatch needs at least one "
                "connected view with its use_* switch on"
            )
        store_device = comfy.model_management.intermediate_device()
        comfy.model_management.load_model_gpu(clip_vision_model.patcher)
        batch_size = views[names[0]].shape[0]
        tok512, tok1024 = {}, {}
        for name in names:
            per512, per1024 = [], []
            for b in range(batch_size):
                frame = views[name][b % views[name].shape[0]]
                bchw = _to_bchw(frame)
                per512.append(
                    _dinov3_encode_global(clip_vision_model, bchw, 512))
                per1024.append(
                    _dinov3_encode_global(clip_vision_model, bchw, 1024))
            tok512[name] = torch.cat(per512, dim=0).to(store_device)
            tok1024[name] = torch.cat(per1024, dim=0).to(store_device)

        pack = {"names": list(names), "tok512": tok512, "tok1024": tok1024,
                "batch": int(batch_size), "axis": front_axis,
                "temp": float(blend_temperature)}
        patched = model.clone()
        patched.set_model_unet_function_wrapper(
            lambda apply_fn, params, _pack=pack: _mv_blend_apply(
                apply_fn, params, _pack)
        )
        logging.info(
            "Trellis2SpatialMultiViewPatch: %d view(s) %s axis=%s temp=%.2f.",
            len(names), "+".join(names), front_axis, blend_temperature,
        )
        return IO.NodeOutput(patched)


def _mv_blend_apply(apply_fn, params, pack):
    """model_function_wrapper: run once per view, blend with spatial weights."""
    try:
        c_in = params["c"]
        x = params["input"]
        t = params["timestep"]
    except KeyError:
        return apply_fn(params["input"], params["timestep"], **params["c"])
    names = pack["names"]
    if len(names) <= 1:
        return apply_fn(x, t, **c_in)
    if "trellis2_proj_feats" in c_in:
        return apply_fn(x, t, **c_in)
    if "embeds" not in c_in or "c_crossattn" not in c_in:
        return apply_fn(x, t, **c_in)

    n_chunks = len(params.get("cond_or_uncond", [])) or 1
    rows = int(x.shape[0])
    if rows % n_chunks != 0:
        logging.warning(
            "Trellis2SpatialMultiViewPatch: rows %d not divisible by %d "
            "chunks, passing through.", rows, n_chunks)
        return apply_fn(x, t, **c_in)
    rpc = rows // n_chunks

    cross = c_in["c_crossattn"]
    embeds = c_in["embeds"]
    closure_batch = pack["batch"]
    if closure_batch != rpc and closure_batch != 1:
        logging.warning(
            "Trellis2SpatialMultiViewPatch: encoded batch %d != runtime "
            "batch %d, passing through.", closure_batch, rpc)
        return apply_fn(x, t, **c_in)

    chunk_rows = int(cross.shape[0]) // n_chunks
    try:
        neg_chunk = [
            bool((cross[o * chunk_rows:(o + 1) * chunk_rows].float()
                  .abs().max().item()) < 1e-6)
            for o in range(n_chunks)
        ]
    except Exception:
        return apply_fn(x, t, **c_in)

    def _fit(tok):
        tok = tok.to(device=cross.device, dtype=cross.dtype)
        if tok.shape[0] == rpc:
            return tok
        return tok[:1].expand(rpc, *tok.shape[1:])

    try:
        outs = []
        for name in names:
            c_view = dict(c_in)
            t512 = _fit(pack["tok512"][name])
            t1024 = _fit(pack["tok1024"][name])
            cross_v = cross.clone()
            embeds_v = (embeds.clone() if torch.is_tensor(embeds)
                        else embeds)
            for o in range(n_chunks):
                if neg_chunk[o]:
                    continue
                s = slice(o * chunk_rows, (o + 1) * chunk_rows)
                cross_v[s] = t512[: cross_v[s].shape[0]]
                if torch.is_tensor(embeds_v):
                    embeds_v[s] = t1024[: embeds_v[s].shape[0]]
            c_view["c_crossattn"] = cross_v
            c_view["embeds"] = embeds_v
            outs.append(apply_fn(x, t, **c_view))
    except Exception as exc:
        logging.warning(
            "Trellis2SpatialMultiViewPatch: per-view pass failed (%s), "
            "passing through.", exc)
        return apply_fn(x, t, **c_in)

    coords = c_in.get("trellis2_coords", None)
    try:
        if coords is not None and torch.is_tensor(coords) and coords.numel():
            return _mv_blend_sparse(outs, coords,
                                    c_in.get("trellis2_coord_counts", None),
                                    names, pack["axis"], pack["temp"])
        return _mv_blend_dense(outs, x.shape, names, pack["axis"],
                               pack["temp"])
    except Exception as exc:
        logging.warning(
            "Trellis2SpatialMultiViewPatch: blend failed (%s), using "
            "first-view output.", exc)
        return outs[0]


def _mv_blend_sparse(outs, coords, counts, views, axis, temp):
    ref = outs[0]
    b = int(ref.shape[0])
    tmax = int(ref.shape[2])
    extra = ref.shape[3:]
    res = int(coords[:, 1:].float().max().item()) + 1
    scores = _mv_view_scores_sparse(coords.long().cpu(), res, views, axis)
    weights = torch.softmax(scores * float(temp), dim=1).to(
        device=ref.device, dtype=torch.float32)
    if counts is not None and torch.is_tensor(counts) and counts.numel():
        counts_list = [int(v) for v in counts.tolist()]
    else:
        counts_list = [coords.shape[0]]
    lb = len(counts_list)
    v = len(views)
    wmap = torch.zeros((b, tmax, v), dtype=torch.float32)
    wmap[..., 0] = 1.0
    for bi in range(b):
        i = bi % lb
        n = min(counts_list[i], tmax)
        off = int(sum(counts_list[:i]))
        wmap[bi, :n] = weights[off: off + n]
    w = (wmap.permute(2, 0, 1).unsqueeze(2).unsqueeze(-1)
         .to(device=ref.device, dtype=ref.dtype))
    stacked = torch.stack(
        [o.reshape(b, o.shape[1], tmax, -1).to(ref.dtype) for o in outs],
        dim=0)
    blended = (stacked * w).sum(dim=0)
    return blended.reshape((b, ref.shape[1], tmax) + tuple(extra))


def _mv_blend_dense(outs, shape, views, axis, temp):
    ref = outs[0]
    b = int(ref.shape[0])
    scores = _mv_view_scores_dense(ref.shape, ref.device, views, axis)
    w = torch.softmax(scores * float(temp), dim=0).to(dtype=ref.dtype)
    stacked = torch.stack([o.to(ref.dtype) for o in outs], dim=0)
    w = w.view(len(views), *([1] * (stacked.ndim - 5)), *ref.shape[2:])
    return (stacked * w.unsqueeze(1)).sum(dim=0)


class Trellis2CFGInterval(IO.ComfyNode):
    """Native port of visualbruno's guidance-interval + guidance-rescale.

    Per step, with sigma read as Bruno's rescaled t (so use together with
    the matching ModelSamplingSD3 shift):
    - inside [start, end]: standard CFG `uncond + cfg*(cond-uncond)`, then
      Bruno's x0 rescale `r*rescaled + (1-r)*cfg` when `rescale > 0`
      (identical math to core RescaleCFG's flow branch, kept here so both
      live in one hook slot).
    - outside: guidance 1.0, pure conditional (matches Bruno's mixin).

    Defaults mirror Bruno's generator widgets: structure/shape
    0.1-1.0 / 0.2, texture 0.0-0.9 / 0.2. Pure torch, no extra deps.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2CFGInterval",
            display_name="Trellis2 CFG Interval + Rescale",
            category="model/conditioning/trellis",
            inputs=[
                IO.Model.Input("model"),
                IO.Float.Input(
                    "interval_start", default=0.1, min=0.0, max=1.0,
                    step=0.01,
                ),
                IO.Float.Input(
                    "interval_end", default=1.0, min=0.0, max=1.0,
                    step=0.01,
                ),
                IO.Float.Input(
                    "guidance_rescale", default=0.2, min=0.0, max=1.0,
                    step=0.01,
                    tooltip="Bruno guidance_rescale (0.2 on every stage).",
                ),
            ],
            outputs=[IO.Model.Output()],
        )

    @classmethod
    def execute(cls, model, interval_start=0.1, interval_end=1.0,
                guidance_rescale=0.2) -> IO.NodeOutput:
        start = float(interval_start)
        end = float(interval_end)
        rescale = float(guidance_rescale)

        def _cfg_fn(args):
            sig = args["sigma"]
            try:
                t = float(sig.reshape(-1)[0].detach().cpu().item())
            except Exception:
                t = float(sig)
            res_c = args["cond"]
            res_u = args["uncond"]
            if not (start <= t <= end):
                return res_c
            cond_scale = args["cond_scale"]
            if rescale > 0:
                x = args["input"]
                d_c = args["cond_denoised"]
                d_u = args["uncond_denoised"]
                d_cfg = d_u + cond_scale * (d_c - d_u)
                dims = tuple(range(1, d_c.ndim))
                std_p = d_c.std(dim=dims, keepdim=True)
                std_g = d_cfg.std(dim=dims, keepdim=True).clamp(min=1e-8)
                d_rs = d_cfg * (std_p / std_g)
                return x - (rescale * d_rs + (1.0 - rescale) * d_cfg)
            return res_u + cond_scale * (res_c - res_u)

        patched = model.clone()
        patched.set_model_sampler_cfg_function(_cfg_fn)
        logging.info(
            "Trellis2CFGInterval: [%.2f, %.2f] rescale=%.2f.",
            start, end, rescale,
        )
        return IO.NodeOutput(patched)


class Trellis2MultiviewExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Trellis2MultiViewConditioning, MeshWithVoxelToNativeBridge, NativeMeshVoxelToMeshWithVoxel, Trellis2SpatialMultiViewPatch, Trellis2CFGInterval]


async def comfy_entrypoint() -> Trellis2MultiviewExtension:
    return Trellis2MultiviewExtension()
