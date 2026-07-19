"use client";

import { useEffect, useRef, useState } from "react";
import { withBasePath } from "@/lib/basePath";

type Baseline = "midas_levit" | "midas_swinv2_t";

const samples = [
  "1",
  "2",
  "3",
  "4",
  "5",
  "DSC01335",
  "DSC01583",
  "DSC01597",
  "DSC01726",
  "IMG_0313",
  "IMG_1040",
  "IMG_1042",
  "IMG_1053-已增强-降噪",
  "IMG_8924-已增强-降噪",
  "IMG_8926-已增强-降噪",
  "IMG_9013-已增强-降噪",
  "IMG_9098",
  "IMG_9174-已增强-降噪",
  "IMG_9853",
  "IMG_9861-已增强-降噪",
  "IMG_9863-已增强-降噪",
  "IMG_9989",
  "monalisa"
];

const baselineLabels: Record<Baseline, string> = {
  midas_levit: "MiDaS v3.1 · LeViT",
  midas_swinv2_t: "MiDaS v3.1 · Swin2-T"
};

function asset(kind: "rgb" | "teacher" | "depthart_l" | Baseline, sample: string) {
  return withBasePath(`/images/self_collection/${kind}/${sample}.webp`);
}

function sampleLabel(sample: string, index: number) {
  if (/^\d+$/.test(sample)) return `Scene ${String(index + 1).padStart(2, "0")}`;
  if (sample === "monalisa") return "Mona Lisa";
  return sample.replace(/-已增强-降噪/g, "");
}

export default function RelativeGallery() {
  const [index, setIndex] = useState(0);
  const [baseline, setBaseline] = useState<Baseline>("midas_levit");
  const [position, setPosition] = useState(50);
  const thumbnails = useRef<Array<HTMLButtonElement | null>>([]);
  const sample = samples[index];

  useEffect(() => {
    thumbnails.current[index]?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    setPosition(50);
  }, [index]);

  useEffect(() => {
    setPosition(50);
  }, [baseline]);

  function move(direction: -1 | 1) {
    setIndex((current) => (current + direction + samples.length) % samples.length);
  }

  return (
    <div className="result-block relative-gallery" aria-labelledby="relative-gallery-title">
      <div className="gallery-heading">
        <div>
          <span className="eyebrow">Self-collected Image</span>
          <h2 id="relative-gallery-title">Relative Depth</h2>
          <span className="gallery-context">{sampleLabel(sample, index)}</span>
        </div>
        <div className="gallery-navigation">
          <span>{String(index + 1).padStart(2, "0")} / {samples.length}</span>
          <button type="button" onClick={() => move(-1)} aria-label="Previous scene">←</button>
          <button type="button" onClick={() => move(1)} aria-label="Next scene">→</button>
        </div>
      </div>

      <div className="gallery-stage">
        <div className="reference-column">
          <figure className="reference-card">
            <figcaption>Input · RGB</figcaption>
            <img src={asset("rgb", sample)} alt={`RGB input for ${sampleLabel(sample, index)}`} />
          </figure>
          <figure className="reference-card">
            <figcaption>Teacher · DepthAnything v2-L</figcaption>
            <img src={asset("teacher", sample)} alt={`DepthAnything v2-L prediction for ${sampleLabel(sample, index)}`} />
          </figure>
        </div>

        <div className="comparison-column">
          <div className="comparison-toolbar">
            <span>Drag to compare</span>
            <div className="baseline-switch" aria-label="MiDaS baseline">
              {(Object.keys(baselineLabels) as Baseline[]).map((option) => (
                <button
                  type="button"
                  key={option}
                  className={baseline === option ? "active" : ""}
                  onClick={() => setBaseline(option)}
                  aria-pressed={baseline === option}
                >{baselineLabels[option]}</button>
              ))}
            </div>
          </div>
          <div className="compare-shell">
            <img className="compare-image" src={asset(baseline, sample)} alt={`${baselineLabels[baseline]} prediction for ${sampleLabel(sample, index)}`} />
            <img className="compare-image compare-image-overlay" style={{ clipPath: `inset(0 0 0 ${position}%)` }} src={asset("depthart_l", sample)} alt={`DepthART-L prediction for ${sampleLabel(sample, index)}`} />
            <div className="compare-divider" style={{ left: `${position}%` }} aria-hidden="true"><span>↔</span></div>
            <input
              className="compare-range"
              type="range"
              min="0"
              max="100"
              value={position}
              onChange={(event) => setPosition(Number(event.target.value))}
              aria-label={`Compare ${baselineLabels[baseline]} with DepthART-L`}
            />
            <span className="compare-label compare-label-left">{baselineLabels[baseline]}</span>
            <span className="compare-label compare-label-right">DepthART-L</span>
          </div>
        </div>
      </div>

      <div className="thumbnail-track" aria-label="Relative depth scenes">
        {samples.map((item, itemIndex) => (
          <button
            type="button"
            key={item}
            ref={(node) => { thumbnails.current[itemIndex] = node; }}
            className={index === itemIndex ? "active" : ""}
            onClick={() => setIndex(itemIndex)}
            aria-label={`Show ${sampleLabel(item, itemIndex)}`}
            aria-pressed={index === itemIndex}
          >
            <img src={asset("rgb", item)} alt="" />
            <span>{String(itemIndex + 1).padStart(2, "0")}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
