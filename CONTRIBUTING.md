# Contributing to gans-awesome-applications

Thanks for helping keep this list awesome! 🎉

The guiding principle of this list is simple:

> **Only awesome is awesome. It's a curation, not a collection.**

Please only suggest work that you (or another contributor) can personally recommend.
When in doubt, leave it out — a tightly curated list is more valuable than an exhaustive one.

## What belongs here

- **GAN _applications_ and demonstrations.** This list is about what GANs are *used for*.
- General-purpose image-generation papers whose contribution is the model itself
  (e.g. plain DCGAN, BEGAN, WGAN) belong in the landmark section at most — not as entries.
- Pure theory / loss-function / training-trick papers are out of scope unless they are
  genuine landmarks the whole community relies on.
- Diffusion-only work is out of scope (this is a GAN list). GAN+diffusion hybrids are fine.

## Quality bar (an entry should meet most of these)

- Notable / well-cited, or a clearly useful working project with code or a live demo.
- Has at least one of: an official paper, official code, a project page, or a runnable demo.
- Is maintained / not abandoned (flag known-dead projects rather than silently adding them).

## Entry format

Add new entries to the **bottom** of the most relevant category, using this exact format:

```
+ Title (Venue Year), [[paper]](url), [[github]](url), [[project]](url), [[demo]](url)
```

Rules:

- **Title** — the paper/project title, optionally with a short alias in parentheses, e.g.
  `... (DragGAN)`.
- **(Venue Year)** — publication venue and year when known, e.g. `(CVPR 2023)`. If you only
  know the year, use `(2023)`. Omit for informal projects/blog posts.
- **Links** — keep this fixed order and only include the ones that exist:
  `[[paper]]` · `[[github]]` (code) · `[[project]]` (project page) · `[[demo]]` / `[[colab]]` ·
  `[[video]]` / `[[youtube]]` · `[[blog]]`.
- Prefer the **official** repo/page. If you must link an unofficial reimplementation, say so.
- One entry per line, starting with `+ `.

## Adding a new category

If no existing category fits, you may add a new `### Category` section. Remember to:

1. Add the section in a sensible place in the body.
2. Add a matching link to the **Contents** table.
3. Add the `back to Contents` link at the end of the new section.

## How to submit

1. Fork the repo and create a branch.
2. Make your change following the format above.
3. Open a pull request with a one-line rationale ("why is this awesome?").

That's it. Thank you for contributing! 🙏
