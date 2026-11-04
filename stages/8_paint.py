#!/usr/bin/env python3
"""Paint saved camera views, upscale their pixels, and bake the fixed-size atlas."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image

PAINT_ATTEMPTS = 3

parser = argparse.ArgumentParser()
parser.add_argument('base', type=Path)
parser.add_argument('reference', type=Path)
parser.add_argument('work', type=Path)
parser.add_argument('out', type=Path)
parser.add_argument('--model', choices=['gpt', 'flux'], default='gpt')
parser.add_argument('--paint-base', choices=['trellis', 'normals'], default='trellis')
parser.add_argument('--texture', type=int, choices=[4096], default=4096)
parser.add_argument('--upscale', type=int, choices=[0, 2, 4], default=2)
parser.add_argument('--texel-density-map', action='store_true')
parser.add_argument('--prompt-file', type=Path)
parser.add_argument('--flux-dir', type=Path)
parser.add_argument('--cameras', type=Path)
parser.add_argument('--paintings', type=Path)
parser.add_argument('--base-material', type=Path)
parser.add_argument('--no-base-colour-match', action='store_true')
parser.add_argument('--colour-reference', type=Path, help='transparent-background image the base is matched to before painting (default: the reference)')
parser.add_argument('--painting-output', type=Path)
parser.add_argument('--phase', choices=['all', 'render', 'paint', 'bake'], default='all',
                    help='render needs the GPU, paint waits on the network, bake needs the GPU; each can hold its own reservation')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
density_tool = root / 'Intermediate/surfaces/atlas-density'
if args.texel_density_map and not density_tool.is_file():
    parser.error('Build the native density tool with tools/build-native.sh first')
python = sys.executable
progressive = args.work / 'progressive' / args.paint_base
views = progressive / 'views'
paintings = args.paintings or args.painting_output or progressive / 'gen'
bake_directory = progressive / 'bake'
progressive.mkdir(parents=True, exist_ok=True)
paintings.mkdir(parents=True, exist_ok=True)
prompt = args.prompt_file or root / 'prompts/progressive_paint.txt'
prompt_text = prompt.read_text().strip()
if not prompt_text:
    parser.error('Painting prompt is empty')
settings = {'base': hashlib.sha256(args.base.read_bytes()).hexdigest(),
            'reference': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            'prompt': prompt_text, 'model': args.model, 'paint_base': args.paint_base,
            'cameras': hashlib.sha256(args.cameras.read_bytes()).hexdigest() if args.cameras else None,
            'paintings': str(paintings.resolve())}
if args.base_material:
    settings['base_material'] = hashlib.sha256(args.base_material.read_bytes()).hexdigest()
match_base_to_reference = args.paint_base == 'trellis' and not args.no_base_colour_match and not args.paintings
if match_base_to_reference:
    settings['base_matched_to_reference'] = True
settings_path = progressive / 'settings.json'
if settings_path.is_file() and json.loads(settings_path.read_text()) != settings:
    parser.error('Painting inputs changed; choose a new output directory to preserve existing views')
settings_path.write_text(json.dumps(settings, indent=2))
(progressive / 'prompt.txt').write_text(prompt_text + '\n')
cameras_path = progressive / 'cameras.json'
if args.cameras and not cameras_path.is_file():
    shutil.copyfile(args.cameras, cameras_path)

def run_tool(name, *arguments, interpreter=python, environment=None, limit=1800):
    print('START', name, flush=True)
    subprocess.run([interpreter, str(root/'tools'/name), *map(str, arguments)],
                   check=True, env=environment, timeout=limit)
    print('DONE', name, flush=True)

material_arguments = ['--base-material', args.base_material] if args.base_material else []
reference_matched_base = progressive / 'base-reference-matched.png'
base_colour_arguments = ['--base-colour', reference_matched_base] if match_base_to_reference else []
def render_paint_inputs(*extra):
    run_tool('render_paint_views.py', args.base, views, cameras_path, '--base', args.paint_base, *material_arguments, *base_colour_arguments, *extra, limit=300)

# The painter receives the base already matched to the reference, so it adds detail rather than correcting colour.
if match_base_to_reference and args.phase != 'paint' and not reference_matched_base.is_file():
    unmatched_views = progressive / 'views-unmatched'
    original_base = progressive / 'base-original.png'
    run_tool('render_paint_views.py', args.base, unmatched_views, cameras_path, '--base', args.paint_base, *material_arguments,
             '--save-base-colour', original_base, limit=300)
    subprocess.run([str(root / 'Intermediate/surfaces/match-base-colour'), str(original_base), str(cameras_path), str(unmatched_views),
                    str(args.colour_reference or args.reference), str(reference_matched_base), str(progressive / 'base-reference-match.toml')], check=True)

if args.phase != 'paint':
    render_paint_inputs()
cameras = json.loads(cameras_path.read_text())
if not 1 <= len(cameras['cams']) <= 16:
    parser.error('The resident baker requires between one and sixteen camera views')
if args.paintings:
    for camera in cameras['cams']:
        painted = paintings / (camera['name'] + '.png')
        if not painted.is_file():
            parser.error(f'Missing saved painting: {painted}')
if args.model == 'gpt':
    sys.path.insert(0, str(root/'tools'))
    from chatgpt_image import CodexImageClient
    from paint_check import measure_painting_against_render
    # Each view is checked once: a painting that used up its attempts isn't requested again on resume.
    checks_path = paintings / 'paint-check.json'
    checks = json.loads(checks_path.read_text()) if checks_path.is_file() else {}
    attempts_directory = progressive / 'attempts'
    attempts_directory.mkdir(exist_ok=True)

    def record_painting_check(name, painted, measured, attempts):
        checks[name] = {**measured, 'attempts': attempts, 'sha256': hashlib.sha256(painted.read_bytes()).hexdigest()}
        checks_path.write_text(json.dumps(checks, indent=2))
        print('PAINT_CHECK', name, f"attempts={attempts}", *(f'{k}={v:.3f}' for k, v in measured.items()), flush=True)

    to_paint = []
    if not args.paintings:
        for camera in cameras['cams']:
            name = camera['name']
            painted = paintings / (name + '.png')
            if not painted.is_file():
                to_paint.append(name)
            elif checks.get(name, {}).get('sha256') != hashlib.sha256(painted.read_bytes()).hexdigest():
                measured = measure_painting_against_render(views / (name + '.png'), painted)
                if measured['score'] <= 1:
                    record_painting_check(name, painted, measured, 0)
                else:
                    print('PAINT_REJECTED', name, f"score={measured['score']:.2f}", flush=True)
                    painted.unlink()
                    to_paint.append(name)
    if to_paint and args.phase in ('all', 'render'):
        render_paint_inputs('--reframe', *to_paint)
        cameras = json.loads(cameras_path.read_text())
        if args.cameras:
            # The saved rig stays paired with the paintings it now produces.
            shutil.copyfile(cameras_path, args.cameras)
            settings['cameras'] = hashlib.sha256(args.cameras.read_bytes()).hexdigest()
            settings_path.write_text(json.dumps(settings, indent=2))
        shutil.rmtree(progressive / 'grade-views', ignore_errors=True)
    if args.phase == 'render':
        sys.exit()
    client = CodexImageClient() if to_paint else None
    for name in to_paint:
        rendered = views / (name + '.png')
        painted = paintings / (name + '.png')
        results = []
        for attempt in range(1, PAINT_ATTEMPTS + 1):
            candidate = attempts_directory / f'{name}-{attempt}.png'
            if not candidate.is_file():
                client.generate(prompt_text, candidate, [rendered, args.reference],
                                size=f"{cameras['res']}x{cameras['res']}")
            results.append((measure_painting_against_render(rendered, candidate), candidate))
            print('PAINT_ATTEMPT', name, attempt, f"score={results[-1][0]['score']:.2f}", flush=True)
            if results[-1][0]['score'] <= 1:
                break
        measured, best = min(results, key=lambda result: result[0]['score'])
        shutil.copyfile(best, painted)
        for _, candidate in results:
            candidate.unlink()
        record_painting_check(name, painted, measured, len(results))
else:
    flux_root = (args.flux_dir or Path(os.environ.get('ESPALIER_FLUX_ROOT', root/'src/flux-3-image-klein'))).resolve()
    flux_python = os.environ.get('ESPALIER_FLUX_PYTHON', str(flux_root/'.venv/bin/python'))
    flux_environment = os.environ.copy()
    flux_environment['PYTHONPATH'] = str(flux_root/'src') + os.pathsep + flux_environment.get('PYTHONPATH', '')
    flux_environment['WEIGHTS_DIR'] = str(flux_root/'weights')
    jobs = progressive / 'flux-jobs.tsv'
    with jobs.open('w', newline='') as destination:
        writer = csv.writer(destination, delimiter='\t')
        for camera in cameras['cams']:
            name = camera['name']
            if not (paintings / (name + '.png')).is_file():
                writer.writerow([progressive / (name + '-caption.txt'), paintings / (name + '.png'),
                                 1024, views / (name + '.png'), args.reference])
    if jobs.stat().st_size:
        run_tool('flux_upsample.py', '--batch', progressive/'prompt.txt', jobs,
                 interpreter=flux_python, environment=flux_environment)
        run_tool('flux_paint.py', '--batch', jobs, interpreter=flux_python, environment=flux_environment)
if args.phase == 'paint':
    sys.exit()
for index, camera in enumerate(cameras['cams']):
    name = camera['name']
    painted = paintings / (name + '.png')
    if not painted.is_file():
        parser.error(f'Missing painting: {painted}')
    with Image.open(painted) as image:
        image.verify()
    with Image.open(painted) as image:
        if image.format != 'PNG' or image.width != image.height:
            parser.error(f'Expected a square PNG painting: {painted}')
    print(f'PAINTED {index + 1}/{len(cameras["cams"])} {name}', flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
projected_paintings = paintings
if args.upscale:
    projected_paintings = progressive / f'gen-upscaled-{args.upscale}x'
    run_tool('upscale.py', paintings, projected_paintings, '--factor', args.upscale, '--cameras', cameras_path,)
grade_arguments = []
if not args.no_base_colour_match:
    grade_views = views if args.paint_base == 'trellis' else progressive / 'grade-views'
    if args.paint_base != 'trellis':
        run_tool('render_paint_views.py', args.base, grade_views, cameras_path, '--base', 'trellis', *material_arguments, limit=300)
    grade_arguments = ['--colour-match-views', grade_views]
run_tool('bake_paint_views.py', args.base, cameras_path, projected_paintings, bake_directory, args.out, *material_arguments, *base_colour_arguments, *grade_arguments, limit=300)
if args.texel_density_map:
    print('START texel density map', flush=True)
    prefix = args.out.with_name(args.out.stem + '-texel-density')
    subprocess.run([str(density_tool), str(bake_directory), str(cameras_path),
                    str(projected_paintings), str(args.out), str(prefix)], check=True, timeout=900)
    print('DONE texel density map', str(prefix) + '.png', flush=True)
print('DETAIL_PAINTING_COMPLETE', args.out, flush=True)
