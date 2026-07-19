"use client";

import { useEffect, useRef, useState } from "react";
import { withBasePath } from "@/lib/basePath";

const scenes = [
  { id: "11", label: "Train Station", environment: "Indoor" },
  { id: "15", label: "Office", environment: "Indoor" },
  { id: "20", label: "Street", environment: "Outdoor" },
  { id: "22", label: "Commercial District", environment: "Outdoor" }
];

export default function IPhoneDepthShowcase() {
  const [active, setActive] = useState(0);
  const depthRef = useRef<HTMLVideoElement>(null);
  const rgbRef = useRef<HTMLVideoElement>(null);
  const scene = scenes[active];

  useEffect(() => {
    const depth = depthRef.current;
    const rgb = rgbRef.current;
    if (!depth || !rgb) return;
    depth.load();
    rgb.load();
    Promise.allSettled([depth.play(), rgb.play()]);
  }, [active]);

  function syncRgb() {
    const depth = depthRef.current;
    const rgb = rgbRef.current;
    if (!depth || !rgb) return;
    if (Math.abs(rgb.currentTime - depth.currentTime) > 0.12) rgb.currentTime = depth.currentTime;
  }

  function playPair() {
    rgbRef.current?.play().catch(() => undefined);
  }

  function pausePair() {
    rgbRef.current?.pause();
  }

  return (
    <section className="iphone-showcase" id="iphone-demo" aria-labelledby="iphone-demo-title">
      <div className="iphone-copy">
        <span className="eyebrow">Videos from ADVIO datasets, captured on iPhone</span>
        <h2 id="iphone-demo-title">Relative Depth in the Wild</h2>
        <p>DepthART processes handheld portrait video while preserving fine boundaries and accurate depth.</p>
        <div className="iphone-scene-tabs" aria-label="iPhone video scenes">
          {scenes.map((item, index) => (
            <button key={item.id} type="button" className={active === index ? "active" : ""} onClick={() => setActive(index)} aria-pressed={active === index}>
              <span>{item.label}</span><small>{item.environment}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="iphone-device-shell">
        <div className="iphone-hardware">
          <div className="iphone-dynamic-island" aria-hidden="true" />
          <video
            key={`depth-${scene.id}`}
            ref={depthRef}
            className="iphone-depth-video"
            src={withBasePath(`/videos/iphone/advio-${scene.id}-depthart.mp4`)}
            autoPlay muted loop playsInline controls preload="metadata"
            onPlay={playPair}
            onPause={pausePair}
            onTimeUpdate={syncRgb}
            onSeeked={syncRgb}
            aria-label={`${scene.label} DepthART-L relative depth`}
          />
          <div className="iphone-rgb-float">
            <span>RGB</span>
            <video
              key={`rgb-${scene.id}`}
              ref={rgbRef}
              src={withBasePath(`/videos/iphone/advio-${scene.id}-rgb.mp4`)}
              autoPlay muted loop playsInline preload="metadata"
              aria-label={`${scene.label} RGB input`}
            />
          </div>
          <div className="iphone-video-badge"><i />DepthART-L · Relative</div>
        </div>
      </div>
    </section>
  );
}
