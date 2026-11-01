"""Expand painting prompts through the FLUX SDK, then release the prompt model."""
import csv
import sys
from pathlib import Path
from PIL import Image
from flux3.pup import ENGINE, Upsampler


def main():
    if sys.argv[1] == '--batch':
        prompt = Path(sys.argv[2]).read_text()
        with open(sys.argv[3]) as source:
            jobs = [(row[0], row[3:]) for row in csv.reader(source, delimiter='\t')]
    else:
        prompt = Path(sys.argv[1]).read_text()
        jobs = [(sys.argv[2], sys.argv[3:])]
    jobs = [(output, references) for output, references in jobs if not Path(output).is_file()]
    if not jobs:
        return
    ENGINE.update(max_num_seqs=1, max_num_batched_tokens=4096, enforce_eager=True,
                  mm_processor_kwargs={'size': {'longest_edge': 724*724, 'shortest_edge': 65536}})
    upsampler = Upsampler(memory=.80, quantization='fp8', max_model_len=4096)
    for index, (output, references) in enumerate(jobs, 1):
        caption = upsampler(prompt, [Image.open(path) for path in references])
        Path(output).write_text(caption + '\n')
        print(f'PROMPT_UPSAMPLED {index}/{len(jobs)} {output}', flush=True)


if __name__ == '__main__':
    main()
