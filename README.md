# ComfyUI-Trellis2-Multiview-Native

Native-style multi-view conditioning for Trellis2, plus an in-memory
MeshWithVoxel to native MESH + VOXEL bridge that carries full PBR color
(base color RGB + metallic + roughness).

No compiled dependencies beyond what ComfyUI already ships with.

## Nodes

### Trellis2 Multi-View Conditioning (`Trellis2MultiViewConditioning`)

Averages DINOv3 global tokens across 1–4 views (front / left / back / right)
and emits the exact same CONDITIONING format as core `Trellis2Conditioning`,
so all downstream core nodes (Trellis2ShapeStage / KSampler /
Trellis2UpsampleStage / Trellis2TextureStage / VAE decodes) work unchanged.

Trellis2 diffusion uses global cross-attention trained single-view, so
mean-pooling the global tokens is the architecturally consistent way to
condition on several views without retraining. View order is irrelevant to
the diffusion; the first connected view is treated as "front" only as a
posing convention. Each view has a `use_*` switch to skip a wired view
without disconnecting it.

### Trellis2 MeshWithVoxel to Native Bridge (`MeshWithVoxelToNativeBridge`)

Converts visualbruno `ComfyUI-Trellis2` MESHWITHVOXEL to native ComfyUI
`MESH` + `VOXEL` in memory — no GLB export, no shape-encoder re-encode,
no VAE needed.

- `mesh` output: feed into MeshReconstructWithQuad / WTiVo / LODTailor.
- `voxel_colors` output: feed into core `BakeTextureFromVoxel` for real
  color/PBR texture bakes.
- `reorient_zup_to_yup` (default on): applies the same R_x(-90) as
  Trellis2MeshWithVoxelToTrimesh "90 degrees", so mesh vertices and voxel
  indices stay aligned in Y-up space.
- If the input has no voxel grid (generator ran with
  `generate_texture_slat` disabled), the mesh still passes through and an
  empty VOXEL is emitted with a warning — the watertight chain completes
  and Bake produces black maps instead of crashing the run.

## Install

Manual: copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

ComfyUI-Manager: install from Git URL
(`https://github.com/saogalaxy/ComfyUI-Trellis2-Multiview-Native`), then restart.

## Example workflows

- `trellis2_multiview_workflow.json` — native multiview Trellis2:
  LoadImage views -> Multi-View Conditioning -> KSamplers ->
  VAE decodes -> PaintMesh -> GLB.
  Needs: `trellis_2_int8_convrot`, shape/texture VAEs,
  `dino_v3_L_naf_fp32`. Restart ComfyUI after installing.
- Bridged PixelArtistry chain (separate file): Trellis2 multiview
  generator -> Remesh -> Simplify -> FillHoles ->
  **MeshWithVoxelToNativeBridge** -> Quad Reconstruct -> MemoryCleaner ->
  WTiVo watertight -> LODTailor -> Unwrap -> BakeTextureFromVoxel
  (voxel_colors from the bridge) -> ApplyTexture -> GLB saves.
  IMPORTANT: the Trellis2 generator must run with
  `generate_texture_slat` enabled, otherwise there is no color grid to
  bridge and bakes come out black.

## Acknowledgements

Inspired by:

- https://github.com/pixelartistry/PixelArtistry-Pixal3D-Multiview
- https://github.com/visualbruno/ComfyUI-Trellis2
