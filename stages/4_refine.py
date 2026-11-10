#!/usr/bin/env python3
"""Stage 4: Refine Mesh. Hi3D v1.0 refines the dense Hunyuan surface against the cutout at a sparse grid resolution.

Each phase holds one model on the GPU and leaves its result in <out>.phases, so a phase can run in its own process and
GPU reservation: condition, sample-0, decode-0, sample-1, decode-1 and so on, one sample and decode per pass.

  python3 stages/4_refine.py <cutout.png> <dense.ply> <out_refined.ply> [--resolution 1024] [--seed 0] [--phase all]
"""
import ctypes, gc
import argparse, os, time
import torch, trimesh
from PIL import Image
import attention  # provides flash_attn to Hi3D
from sparc3d import Sparc3DPipeline
from sparc3d.nn.sparse.basic import SparseTensor
from sparc3d.pipelines.sparc3d_pipeline import OUTPUT_RESOLUTIONS, PASS_BUDGETS, STRUCTURE_RESOLUTIONS, TARGET_FACE_COUNTS
from sparc3d.sparcubes.postprocess import validate_mesh
from sparc3d.utils.image import prepare_image
from jobs import run_jobs

p = argparse.ArgumentParser()
p.add_argument("cutout", nargs="?", help="input cutout PNG")
p.add_argument("dense", nargs="?", help="input dense mesh (.ply)")
p.add_argument("out", nargs="?", help="output refined mesh (.ply)")
p.add_argument("--batch", help="tab-separated table of the positional paths, holding models across its rows")
p.add_argument("--resolution", type=int, choices=[512, 1024, 1536], default=1024, help="sparse grid resolution (default: 1024)")
p.add_argument("--steps", type=int, default=25, help="scheduler updates per pass (default: 25)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--phase", default="all", help="all, condition, sample-N or decode-N")
a = p.parse_args()

print(f"== 4 refine (Hi3D v1.0, {a.resolution}, phase {a.phase})", flush=True)

device = torch.device("cuda")
def load_pipeline():
    loaded = Sparc3DPipeline.from_pretrained(os.environ.get("ESPALIER_REFINE_MODEL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "Hi3Dv1.0", "weights")))
    loaded._device = device
    return loaded


pipeline = load_pipeline()
passes = OUTPUT_RESOLUTIONS.index(a.resolution) + 1


def report(phase, started):
    print(f"phase {phase}: {time.time() - started:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def evict_weights_from_page_cache():
    """The weights just read fill the page cache, and the final decode's large allocations would otherwise stall reclaiming them."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for name in ("models--hi3d-official--Hi3Dv1.0", "models--tencent--Hunyuan3D-2.1", "models--microsoft--TRELLIS.2-4B", "models--briaai--RMBG-2.0", "models--facebook--dinov3-vitl16-pretrain-lvd1689m"):
        blobs = os.path.join(hub, name, "blobs")
        for entry in os.scandir(blobs) if os.path.isdir(blobs) else ():
            descriptor = os.open(entry.path, os.O_RDONLY)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(descriptor)


def release(*models):
    """Each pass holds tens of gigabytes of system memory at its peak, and earlier models aren't used again for this object."""
    for name in models:
        setattr(pipeline, name, None)
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    evict_weights_from_page_cache()


def condition_phase(cutout, dense_path, state):
    pipeline.conditioner.to(device)
    condition = pipeline.conditioner(prepare_image(Image.open(cutout)))
    pipeline.conditioner.to("cpu")
    coords = pipeline._adaptive_voxelize(validate_mesh(trimesh.load(dense_path, force="mesh", process=False)), STRUCTURE_RESOLUTIONS[0], PASS_BUDGETS[0])
    torch.save(condition.cpu(), f"{state}/condition.pt")
    torch.save(coords.cpu(), f"{state}/coords-0.pt")


@torch.inference_mode()
def sample_phase(index, state):
    structure = STRUCTURE_RESOLUTIONS[index]
    transformer = pipeline.transformer_512 if structure == 64 else pipeline.transformer_1024
    transformer.to(device)
    generator = torch.Generator(device="cpu").manual_seed(a.seed + index)
    latents = pipeline._sample_latents(transformer, torch.load(f"{state}/condition.pt").to(device), torch.load(f"{state}/coords-{index}.pt"), structure,
                                       a.steps, generator, pass_index=index, num_passes=passes)
    transformer.to("cpu")
    torch.save({"feats": latents.feats.cpu(), "coords": latents.coords.cpu()}, f"{state}/latents-{index}.pt")


@torch.inference_mode()
def decode_phase(index, dense_path, state, out):
    latents = torch.load(f"{state}/latents-{index}.pt")
    pipeline.vae.to(device)
    decoded = pipeline._decode_latents(SparseTensor(feats=latents["feats"].to(device), coords=latents["coords"].to(device)), STRUCTURE_RESOLUTIONS[index])
    pipeline.vae.to("cpu")
    mesh = pipeline._extract_mesh(decoded)
    del decoded
    torch.cuda.empty_cache()
    if index + 1 < passes:
        torch.save(pipeline._adaptive_voxelize(mesh, STRUCTURE_RESOLUTIONS[index + 1], PASS_BUDGETS[index + 1]).cpu(), f"{state}/coords-{index + 1}.pt")
        return
    mesh = pipeline._postprocess_mesh(mesh, target_face_count=TARGET_FACE_COUNTS[index])
    # Hi3D returns its own normalised frame. The later stages and any saved cameras are in the dense mesh's.
    dense = trimesh.load(dense_path, force="mesh", process=False)
    scale = dense.extents.max() / mesh.extents.max()
    mesh.vertices = (mesh.vertices - mesh.bounds.mean(0)) * scale + dense.bounds.mean(0)
    mesh.export(out)
    print(f"refined to {len(mesh.faces)} faces -> {out}", flush=True)


def phase_result_exists(phase, state):
    kind, _, index = phase.partition("-")
    files = {"condition": ["condition.pt", "coords-0.pt"], "sample": [f"latents-{index}.pt"], "decode": [f"coords-{int(index) + 1}.pt"] if index and int(index) + 1 < passes else []}[kind]
    return bool(files) and all(os.path.isfile(f"{state}/{name}") for name in files)


def refine_mesh(cutout, dense_path, out):
    state = out + ".phases"
    os.makedirs(state, exist_ok=True)
    phases = ["condition"] + [f"{kind}-{index}" for index in range(passes) for kind in ("sample", "decode")]
    global pipeline
    if pipeline.conditioner is None:
        del pipeline
        gc.collect()
        pipeline = load_pipeline()
    for phase in phases if a.phase == "all" else [a.phase]:
        started = time.time()
        # A phase's result is kept, so an interrupted object carries on from where it stopped.
        if a.phase == "all" and phase_result_exists(phase, state):
            continue
        if phase == f"sample-{passes - 1}" and passes > 1:
            release("conditioner", "transformer_512")
        if phase == f"decode-{passes - 1}":
            release("transformer_512", "conditioner", "_transformer_1024", "_transformer_1024_loader")
        torch.cuda.reset_peak_memory_stats()
        if phase == "condition":
            condition_phase(cutout, dense_path, state)
        elif phase.startswith("sample-"):
            sample_phase(int(phase[7:]), state)
        else:
            decode_phase(int(phase[7:]), dense_path, state, out)
        report(phase, started)


run_jobs(a.batch, (a.cutout, a.dense, a.out), refine_mesh)
