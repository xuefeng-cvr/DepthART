# DepthART Project Page

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

The static export is written to `out/`. The workflow in
`.github/workflows/deploy.yml` builds and publishes it to GitHub Pages whenever
`main` is updated.

## Content

- Page implementation: `src/`
- Images, videos, paper, and benchmark CSV files: `public/`
- GitHub Pages configuration: `next.config.js`

Large videos are intentionally lazy-loaded or require user interaction where
appropriate so that the initial page remains responsive.
