---
name: image-generation
description: Reference and client scripts for generating and editing images from the command line with GPT image models through a ChatGPT plan Codex login.
---

# Image generation

GPT image models can be invoked from the command line using the client in this skill:

```bash
python3 skills/image-generation/scripts/chatgpt_image.py PROMPT_FILE OUT.png [IMAGE ...] [--size WxH]
```

- **Generations vs Edits**: Without input images, the model generates from the prompt. Supplying input images makes it an edit: pass the subject image being edited first, followed by reference images.
- **Output**: Writes the output PNG and prints the output path, elapsed time, and served quality.
- **Authentication**: Uses the Codex login in `~/.codex/auth.json` (or `CODEX_AUTH_PATH`).

## Prompting Guidelines

Every prompt should describe the target visual result concisely:

- **Describe what is present**: State the desired objects, lighting, and materials directly rather than listing exclusions.
- **Consistent structure**: When painting textures over 3D renders, specify the lighting, silhouette adherence, and surface properties clearly (e.g. "flat even lighting, no shadows or specular highlights, pure solid background").
- **Surface detail**: Name the concrete real-world materials and wear patterns to introduce (wood grain, clay pores, metal wear, weave).
- **Text exclusion**: To prevent models from inserting synthetic lettering or watermarks, conclude prompts with "No text, labels, or borders."
