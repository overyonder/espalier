"""Paint camera views through the FLUX SDK, sharing one model load per batch."""
import csv
import sys
import time
from pathlib import Path
from PIL import Image
import torch
from flux3.pipeline import MODELS, WEIGHTS, Pipeline, dit_path, prepare_multi_reference


def main():
    if sys.argv[1] == '--batch':
        with open(sys.argv[2]) as source:
            jobs = list(csv.reader(source, delimiter='\t'))
    else:
        jobs = [sys.argv[1:]]
    shared = WEIGHTS / MODELS['distilled']['dit']
    started = time.monotonic()
    pipeline = Pipeline({'distilled': str(dit_path('distilled', 'nvfp4'))},
                        str(shared / 'text_encoder'), str(shared / 'ae.safetensors'), offload=True)
    print(f'FLUX_LOADED {time.monotonic() - started:.1f}s', flush=True)
    for index, (caption, output, size, *references) in enumerate(jobs, 1):
        started = time.monotonic()
        image = pipeline.generate('distilled', Path(caption).read_text().strip(),
                                  [prepare_multi_reference(Image.open(path)) for path in references],
                                  int(size), int(size), MODELS['distilled']['steps'], None, seed=0)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix('.pending.png')
        image.save(pending)
        pending.replace(destination)
        print(f'FLUX_PAINTED {index}/{len(jobs)} {destination} {time.monotonic() - started:.1f}s', flush=True)
    print(f'FLUX_PEAK_GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
