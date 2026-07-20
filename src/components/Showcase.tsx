"use client";

import type { ReactNode } from "react";
import { withBasePath } from "@/lib/basePath";
import BenchmarkExplorer from "@/components/BenchmarkExplorer";
import RelativeGallery from "@/components/RelativeGallery";
import RelativeVideoGallery from "@/components/RelativeVideoGallery";
import IPhoneDepthShowcase from "@/components/IPhoneDepthShowcase";
import MetricPointCloud from "@/components/MetricPointCloud";

const paper = (name: string) => withBasePath(`/images/paper/${name}`);

function Icon({ children }: { children: ReactNode }) {
  return <span aria-hidden="true">{children}</span>;
}

function PaperIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></svg>;
}

function GitHubIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" stroke="none" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.52-1.34-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.19-1.49 3.15-1.18 3.15-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.07.79 2.16v3.2c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" /></svg>;
}

export default function Showcase() {
  return (
    <main className="site-shell">
      <header className="site-header">
        <nav className="nav page-width" aria-label="Primary navigation">
          <a className="brand" href="#home">Depth<span>ART</span></a>
          <div className="nav-links">
            <a href="#home">Home</a>
            <a href="#results">Results</a>
            <a href="#method">Method</a>
            <a href="#metric-point">3D Demo</a>
            <a href="#citation">BibTeX</a>
          </div>
          <a className="nav-cta" href={withBasePath("/DepthART-paper.pdf")} target="_blank" rel="noreferrer">
            <Icon>↗</Icon> Paper
          </a>
        </nav>
      </header>

      <section className="hero page-width" id="home">
        <div className="hero-centered">
          <p className="hero-expansion">Depth <span>A</span>nything <span>R</span>ethought for <span>T</span>iny Models</p>
          <div className="hero-title-line"><h1>Depth<span className="title-accent">ART</span></h1><span className="hero-venue-tag">ACMMM 2026</span></div>
          <p className="hero-kicker">Scaling Foundation Monocular Depth to Tiny Models</p>
          <div className="hero-actions">
            <a className="button-dark" href={withBasePath("/DepthART-paper.pdf")} target="_blank" rel="noreferrer"><PaperIcon /> Read paper</a>
            <a className="button-light" href="https://github.com/xuefeng-cvr/DepthART" target="_blank" rel="noreferrer"><GitHubIcon /> Code</a>
            <a className="button-light" href="https://huggingface.co/Fengxue93/DepthART/tree/main" target="_blank" rel="noreferrer"><Icon>🤗</Icon> Models</a>
          </div>
        </div>
        <div className="kpi-strip" aria-label="DepthART highlights">
          {[
            ["6.0M / 8.0M", "Relative-S / Metric-S parameters", "◫"],
            ["0.92 ms", "DepthART-S 224 · A6000 TensorRT FP16", "◷"],
            ["1088.9", "DepthART-S 224 · A6000 FPS · TensorRT FP16", "↯"],
            ["247.8 / 15.2", "Orin NX TensorRT FP16 / Nano FP32 FPS", "▧"],
            ["0.971 δ1", "DepthART-L · NYUD v2 zero-shot", "◎"]
          ].map(([value, label, icon]) => (
            <div className="kpi" key={label}><div className="kpi-icon">{icon}</div><div><strong>{value}</strong><span>{label}</span></div></div>
          ))}
        </div>
      </section>

      <div className="content-stack page-width">
        <IPhoneDepthShowcase />

        <section className="section-card algorithm-intro" id="method" aria-labelledby="algorithm-overview-title">
          <span className="eyebrow">Algorithm Overview</span>
          <h2 id="algorithm-overview-title">DepthART</h2>
          <p className="algorithm-copy">Recent geometric foundation models have advanced monocular depth estimation, yet their benefits remain limited for tiny models. We present <strong><em>DepthART</em></strong>, a compact model designed for robust on-device depth estimation across diverse scenes. To address dataset-specific overfitting and unstable metric adaptation under camera shifts, DepthART combines bias-resistant data sampling with camera-conditioned fine-tuning that preserves the distilled encoder while adapting metric scale using camera intrinsics. These designs improve both cross-dataset generalization and metric depth prediction in capacity-constrained models.</p>
          <div className="method-grid algorithm-method-grid">
            <article className="method-card"><div className="method-meta"><div className="method-index">1</div><div><strong>Bias-resistant distillation</strong><span>Rebalance a 44M multi-source corpus, then distill DepthAnything v2-L.</span></div></div><video className="novelty-video" src={withBasePath("/videos/brds.mp4")} poster={paper("brds.webp")} autoPlay muted loop playsInline preload="metadata" aria-label="Animated bias-resistant data sampling and distillation pipeline" /></article>
            <article className="method-card"><div className="method-meta"><div className="method-index">2</div><div><strong>Camera-conditioned fine-tuning</strong><span>Freeze the trunk and adapt scale using camera prompts and a multi-query head.</span></div></div><video className="novelty-video" src={withBasePath("/videos/camft.mp4")} poster={paper("camft.webp")} autoPlay muted loop playsInline preload="metadata" aria-label="Animated camera-conditioned fine-tuning architecture" /></article>
          </div>
        </section>

        <BenchmarkExplorer />

        <section className="section-card result-stack">
          <RelativeGallery />
        </section>

        <RelativeVideoGallery />

        <MetricPointCloud />

        <section className="section-card citation" id="citation">
          <div className="citation-grid"><div><span className="eyebrow">Cite this work</span><h2>DepthART</h2><p>Accepted to ACM Multimedia 2026. The final citation and public model links will be added with the camera-ready release.</p><div className="hero-actions"><a className="button-dark" href={withBasePath("/DepthART-paper.pdf")} target="_blank" rel="noreferrer">Read manuscript</a><a className="button-light" href="https://github.com/xuefeng-cvr/DepthART" target="_blank" rel="noreferrer">GitHub repository</a></div></div><pre>{`@inproceedings{depthart2026,\n  title = {DepthART: Scaling Foundation Monocular Depth to Tiny Models},\n  author = {Feng Xue and Wu Chen and Mingshuai Zhao and Guofeng Zhong and Anlong Ming and Haozhe Wang and Dianqiao Lei and Zhaowen Lin and Haiyang Zhang and Nicu Sebe},\n  booktitle = {ACM Multimedia},\n  year = {2026}\n}`}</pre></div>
        </section>
      </div>

      <footer className="footer"><div className="footer-inner page-width"><span>DepthART · Scaling Foundation Monocular Depth to Tiny Models</span><span>Accepted at ACM Multimedia 2026</span></div></footer>
    </main>
  );
}
