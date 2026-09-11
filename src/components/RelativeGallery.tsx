"use client";

import { useEffect, useRef, useState } from "react";
import { withBasePath } from "@/lib/basePath";

type Baseline =
  | "midas_levit"
  | "midas_swinv2_t"
  | "yolo26n"
  | "yolo26s"
  | "yolo26m"
  | "yolo26l"
  | "yolo26x"
  | "zipdepth";

type DepthARTModel =
  | "depthart_tinyvim_s"
  | "depthart_tinyvim_b"
  | "depthart_tinyvim_l"
  | "depthart_mnv4m"
  | "depthart_mnv4mslim_spf";

type Scene = { sample: string; prediction: string };

const scenes: Scene[] = [
  { sample: "1", prediction: "001_1.jpg" },
  { sample: "2", prediction: "002_2.jpg" },
  { sample: "3", prediction: "003_3.jpg" },
  { sample: "4", prediction: "004_4.jpg" },
  { sample: "5", prediction: "005_5.jpg" },
  { sample: "DSC01335", prediction: "007_DSC01335.jpg" },
  { sample: "DSC01583", prediction: "009_DSC01583.jpg" },
  { sample: "DSC01597", prediction: "010_DSC01597.jpg" },
  { sample: "DSC01726", prediction: "012_DSC01726.jpg" },
  { sample: "IMG_0313", prediction: "013_IMG_0313.jpg" },
  { sample: "IMG_1040", prediction: "016_IMG_1040.jpg" },
  { sample: "IMG_1042", prediction: "017_IMG_1042.jpg" },
  { sample: "IMG_1053-已增强-降噪", prediction: "018_IMG_1053-已增强-降噪.jpg" },
  { sample: "IMG_8924-已增强-降噪", prediction: "019_IMG_8924-已增强-降噪.jpg" },
  { sample: "IMG_8926-已增强-降噪", prediction: "020_IMG_8926-已增强-降噪.jpg" },
  { sample: "IMG_9013-已增强-降噪", prediction: "022_IMG_9013-已增强-降噪.jpg" },
  { sample: "IMG_9098", prediction: "023_IMG_9098.jpg" },
  { sample: "IMG_9174-已增强-降噪", prediction: "024_IMG_9174-已增强-降噪.jpg" },
  { sample: "IMG_9853", prediction: "025_IMG_9853.jpg" },
  { sample: "IMG_9861-已增强-降噪", prediction: "026_IMG_9861-已增强-降噪.jpg" },
  { sample: "IMG_9863-已增强-降噪", prediction: "027_IMG_9863-已增强-降噪.jpg" },
  { sample: "IMG_9989", prediction: "028_IMG_9989.jpg" },
  { sample: "monalisa", prediction: "029_monalisa.jpg" }
];

const baselineLabels: Record<Baseline, { full: string; short: string }> = {
  midas_levit: { full: "MiDaS v3.1 · LeViT", short: "MiDaS LeViT" },
  midas_swinv2_t: { full: "MiDaS v3.1 · Swin2-T", short: "MiDaS Swin2-T" },
  yolo26n: { full: "YOLO26-N-Depth", short: "YOLO26-N" },
  yolo26s: { full: "YOLO26-S-Depth", short: "YOLO26-S" },
  yolo26m: { full: "YOLO26-M-Depth", short: "YOLO26-M" },
  yolo26l: { full: "YOLO26-L-Depth", short: "YOLO26-L" },
  yolo26x: { full: "YOLO26-X-Depth", short: "YOLO26-X" },
  zipdepth: { full: "ZipDepth Base", short: "ZipDepth" }
};

const depthartLabels: Record<DepthARTModel, { full: string; short: string }> = {
  depthart_tinyvim_s: { full: "DepthART TinyViM-S · 448", short: "TinyViM-S" },
  depthart_tinyvim_b: { full: "DepthART TinyViM-B · 448", short: "TinyViM-B" },
  depthart_tinyvim_l: { full: "DepthART TinyViM-L · 448", short: "TinyViM-L" },
  depthart_mnv4m: { full: "DepthART MobileNetV4-M · 448", short: "MNV4-M" },
  depthart_mnv4mslim_spf: { full: "DepthART MobileNetV4-M-Slim-SPF · 448", short: "MNV4-M-Slim-SPF" }
};

const baselines = Object.keys(baselineLabels) as Baseline[];
const depthartModels = Object.keys(depthartLabels) as DepthARTModel[];

function referenceAsset(kind: "rgb" | "teacher", sample: string) {
  return withBasePath(`/images/self_collection/${kind}/${sample}.webp`);
}

function comparisonAsset(method: Baseline | DepthARTModel, scene: Scene) {
  if (method === "midas_levit" || method === "midas_swinv2_t") {
    return withBasePath(`/images/self_collection/${method}/${scene.sample}.webp`);
  }
  return withBasePath(`/images/self_comparison/${method}/${scene.prediction}`);
}

function baselineFamily(method: Baseline) {
  if (method.startsWith("midas")) return "midas";
  if (method.startsWith("yolo")) return "yolo";
  return "zipdepth";
}

function sceneLabel(scene: Scene, index: number) {
  if (/^\d+$/.test(scene.sample)) return `Scene ${String(index + 1).padStart(2, "0")}`;
  if (scene.sample === "monalisa") return "Mona Lisa";
  return scene.sample.replace(/-已增强-降噪/g, "");
}

export default function RelativeGallery() {
  const [index, setIndex] = useState(0);
  const [baseline, setBaseline] = useState<Baseline>("yolo26l");
  const [depthartModel, setDepthartModel] = useState<DepthARTModel>("depthart_mnv4mslim_spf");
  const [position, setPosition] = useState(50);
  const thumbnails = useRef<Array<HTMLButtonElement | null>>([]);
  const scene = scenes[index];

  useEffect(() => {
    const thumbnail = thumbnails.current[index];
    const track = thumbnail?.parentElement;
    if (thumbnail && track) {
      const thumbnailRect = thumbnail.getBoundingClientRect();
      const trackRect = track.getBoundingClientRect();
      track.scrollTo({
        left: track.scrollLeft + thumbnailRect.left - trackRect.left - (track.clientWidth - thumbnailRect.width) / 2,
        behavior: "smooth"
      });
    }
    setPosition(50);
  }, [index]);

  useEffect(() => {
    setPosition(50);
  }, [baseline, depthartModel]);

  function move(direction: -1 | 1) {
    setIndex((current) => (current + direction + scenes.length) % scenes.length);
  }

  return (
    <div className="result-block relative-gallery" id="self-collected" aria-labelledby="relative-gallery-title">
      <div className="gallery-heading">
        <div>
          <span className="eyebrow">Self-collected Image</span>
          <h2 id="relative-gallery-title">Relative Depth</h2>
          <span className="gallery-context">{sceneLabel(scene, index)}</span>
        </div>
        <div className="gallery-navigation">
          <span>{String(index + 1).padStart(2, "0")} / {scenes.length}</span>
          <button type="button" onClick={() => move(-1)} aria-label="Previous scene">←</button>
          <button type="button" onClick={() => move(1)} aria-label="Next scene">→</button>
        </div>
      </div>

      <div className="gallery-stage self-gallery-stage">
        <div className="reference-column">
          <figure className="reference-card">
            <figcaption>Input · RGB</figcaption>
            <img src={referenceAsset("rgb", scene.sample)} alt={`RGB input for ${sceneLabel(scene, index)}`} />
          </figure>
          <figure className="reference-card">
            <figcaption>Teacher · DepthAnything v2-L</figcaption>
            <img src={referenceAsset("teacher", scene.sample)} alt={`DepthAnything v2-L prediction for ${sceneLabel(scene, index)}`} />
          </figure>
        </div>

        <div className="comparison-column">
          <div className="compare-method-controls self-gallery-method-controls">
            <div className="method-switch">
              <span>Left · Comparison method</span>
              <div className="baseline-switch" aria-label="Comparison method shown on the left">
                {baselines.map((option) => (
                  <button type="button" key={option} className={`family-${baselineFamily(option)} ${baseline === option ? "active" : ""}`} onClick={() => setBaseline(option)} aria-pressed={baseline === option}>{baselineLabels[option].short}</button>
                ))}
              </div>
            </div>
            <div className="method-switch method-switch-ours">
              <span>Right · DepthART</span>
              <div className="baseline-switch" aria-label="DepthART method shown on the right">
                {depthartModels.map((option) => (
                  <button type="button" key={option} className={depthartModel === option ? "active" : ""} onClick={() => setDepthartModel(option)} aria-pressed={depthartModel === option}>{depthartLabels[option].short}</button>
                ))}
              </div>
            </div>
          </div>

          <div className="compare-shell self-gallery-compare-shell">
            <img className="compare-image" src={comparisonAsset(baseline, scene)} alt={`${baselineLabels[baseline].full} prediction for ${sceneLabel(scene, index)}`} />
            <img className="compare-image compare-image-overlay" style={{ clipPath: `inset(0 0 0 ${position}%)` }} src={comparisonAsset(depthartModel, scene)} alt={`${depthartLabels[depthartModel].full} prediction for ${sceneLabel(scene, index)}`} />
            <div className="compare-divider" style={{ left: `${position}%` }} aria-hidden="true"><span>↔</span></div>
            <input className="compare-range" type="range" min="0" max="100" value={position} onChange={(event) => setPosition(Number(event.target.value))} aria-label={`Compare ${baselineLabels[baseline].full} with ${depthartLabels[depthartModel].full}`} />
            <span className="compare-label compare-label-left">{baselineLabels[baseline].full}</span>
            <span className="compare-label compare-label-right">{depthartLabels[depthartModel].full}</span>
          </div>
        </div>
      </div>

      <div className="thumbnail-track" aria-label="Self-collected relative-depth scenes">
        {scenes.map((item, itemIndex) => (
          <button type="button" key={item.sample} ref={(node) => { thumbnails.current[itemIndex] = node; }} className={index === itemIndex ? "active" : ""} onClick={() => setIndex(itemIndex)} aria-label={`Show ${sceneLabel(item, itemIndex)}`} aria-pressed={index === itemIndex}>
            <img loading="lazy" src={referenceAsset("rgb", item.sample)} alt="" />
            <span>{String(itemIndex + 1).padStart(2, "0")}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
