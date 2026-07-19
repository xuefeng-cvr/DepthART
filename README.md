# DepthART Project Page Source

Official project page for **DepthART: Depth Anything Rethought for Tiny Models**.

## Local development

```bash
npm install
npm run dev
```

The site uses `/DepthART` as its default base path, matching the GitHub Pages URL.

## Static build

```bash
npm run build
```

The static export is written to `out/`. The editable project-page source lives
on the `website` branch, while GitHub Pages serves the compiled contents of the
`gh-pages` branch. Pushes to `website` run a build check through
`.github/workflows/deploy.yml`; publishing remains a separate update to
`gh-pages`.

## Content

- Page implementation: `src/`
- Images, videos, paper, and benchmark CSV files: `public/`
- GitHub Pages configuration: `next.config.js`

Large videos are intentionally lazy-loaded or require user interaction where
appropriate so that the initial page remains responsive.
