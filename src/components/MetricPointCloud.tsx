"use client";

import { useEffect, useRef, useState } from "react";
import { withBasePath } from "@/lib/basePath";

const scenes = [
  "cafe1-1", "cafe1-2",
  "corridor1-1", "corridor1-2", "corridor1-4",
  "home1-2", "home1-3", "home1-4",
  "market1-1", "market1-2", "market1-3",
  "office1-3", "office1-4", "office1-6"
];

function sceneTitle(id: string) {
  const [place, take] = id.split("-");
  const placeName = place.replace(/1$/, "");
  return `${placeName.charAt(0).toUpperCase()}${placeName.slice(1)} · Scene ${take}`;
}

export default function MetricPointCloud() {
  const [active, setActive] = useState(0);
  const [started, setStarted] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const id = scenes[active];

  useEffect(() => {
    videoRef.current?.pause();
    setStarted(false);
  }, [active]);

  function play() {
    const video = videoRef.current;
    if (!video) return;
    setStarted(true);
    video.play().catch(() => setStarted(false));
  }

  return (
    <section className="section-card metric-point-section" id="metric-point" aria-labelledby="metric-point-title">
      <div className="metric-point-heading">
        <div><span className="eyebrow">Robot deployment</span><h2 id="metric-point-title">Metric Point Cloud</h2></div>
        <div className="nyud-only-badge"><strong>DepthART-Metric-L</strong><span>Fine-tuned on NYUD v2 only</span></div>
      </div>

      <div className="metric-point-stage">
        <div className="metric-video-title"><strong>{sceneTitle(id)}</strong><span>Metric 3D reconstruction</span></div>
        <video
          key={id}
          ref={videoRef}
          src={withBasePath(`/videos/metric-point/${id}.mp4`)}
          poster={withBasePath(`/images/metric-point-posters/${id}.jpg`)}
          preload="none"
          playsInline
          controls={started}
          onEnded={() => setStarted(false)}
          aria-label={`${sceneTitle(id)} metric point cloud reconstruction`}
        />
        {!started && <button type="button" className="metric-play" onClick={play} aria-label={`Play ${sceneTitle(id)}`}><span>▶</span><small>Play scene</small></button>}
      </div>

      <div className="metric-scene-track" aria-label="Metric point cloud scenes">
        {scenes.map((scene, index) => (
          <button key={scene} type="button" className={active === index ? "active" : ""} onClick={() => setActive(index)} aria-pressed={active === index}>
            <img src={withBasePath(`/images/metric-point-posters/${scene}.jpg`)} alt="" loading="lazy" />
            <span>{sceneTitle(scene)}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
