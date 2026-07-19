"use client";

import { useEffect, useRef, useState } from "react";
import { withBasePath } from "@/lib/basePath";

const samples = [
  { id: "busy-street-in-the-city", label: "Busy city street" },
  { id: "flower", label: "Flower" },
  { id: "group-of-friends-partying-happily", label: "Friends at a party" },
  { id: "parrot", label: "Parrot" },
  { id: "students-walking-in-a-university", label: "University campus" },
  { id: "woman", label: "Portrait" }
];

function videoAsset(id: string, kind: "rgb" | "depthart") {
  return withBasePath(`/videos/relative/${id}-${kind}.mp4`);
}

function posterAsset(id: string) {
  return withBasePath(`/images/video_posters/${id}.webp`);
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

export default function RelativeVideoGallery() {
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const rgbRef = useRef<HTMLVideoElement>(null);
  const depthRef = useRef<HTMLVideoElement>(null);
  const thumbnails = useRef<Array<HTMLButtonElement | null>>([]);
  const sample = samples[index];

  useEffect(() => {
    const rgb = rgbRef.current;
    const depth = depthRef.current;
    if (!rgb || !depth) return;
    rgb.load();
    depth.load();
    rgb.currentTime = 0;
    depth.currentTime = 0;
    setCurrentTime(0);
    setDuration(0);
    const start = window.setTimeout(() => {
      Promise.allSettled([rgb.play(), depth.play()]).then(() => setIsPlaying(!rgb.paused));
    }, 80);
    thumbnails.current[index]?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    return () => window.clearTimeout(start);
  }, [index]);

  function move(direction: -1 | 1) {
    setIndex((current) => (current + direction + samples.length) % samples.length);
  }

  function togglePlayback() {
    const rgb = rgbRef.current;
    const depth = depthRef.current;
    if (!rgb || !depth) return;
    if (rgb.paused) {
      depth.currentTime = rgb.currentTime;
      Promise.allSettled([rgb.play(), depth.play()]).then(() => setIsPlaying(true));
    } else {
      rgb.pause();
      depth.pause();
      setIsPlaying(false);
    }
  }

  function syncFromRgb() {
    const rgb = rgbRef.current;
    const depth = depthRef.current;
    if (!rgb || !depth) return;
    setCurrentTime(rgb.currentTime);
    if (Math.abs(depth.currentTime - rgb.currentTime) > 0.08) depth.currentTime = rgb.currentTime;
    if (!rgb.paused && depth.paused && depth.readyState >= 3) void depth.play();
  }

  function seek(time: number) {
    const rgb = rgbRef.current;
    const depth = depthRef.current;
    if (!rgb || !depth) return;
    rgb.currentTime = time;
    depth.currentTime = time;
    setCurrentTime(time);
  }

  return (
    <section className="section-card video-gallery-section" aria-labelledby="relative-video-title">
      <div className="gallery-heading">
        <div>
          <span className="eyebrow">Web Video</span>
          <h2 id="relative-video-title">Relative Depth</h2>
          <span className="gallery-context">{sample.label}</span>
        </div>
        <div className="gallery-navigation">
          <span>{String(index + 1).padStart(2, "0")} / {samples.length}</span>
          <button type="button" onClick={() => move(-1)} aria-label="Previous video">←</button>
          <button type="button" onClick={() => move(1)} aria-label="Next video">→</button>
        </div>
      </div>

      <div className="synced-video-stage">
        <figure className="video-panel">
          <figcaption>Input · RGB video</figcaption>
          <video
            ref={rgbRef}
            src={videoAsset(sample.id, "rgb")}
            poster={posterAsset(sample.id)}
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            disablePictureInPicture
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
            onTimeUpdate={syncFromRgb}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
        </figure>
        <figure className="video-panel">
          <figcaption>Prediction · DepthART-L</figcaption>
          <video
            ref={depthRef}
            src={videoAsset(sample.id, "depthart")}
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            disablePictureInPicture
          />
        </figure>
      </div>

      <div className="shared-video-controls">
        <button type="button" className="video-play" onClick={togglePlayback} aria-label={isPlaying ? "Pause both videos" : "Play both videos"}>{isPlaying ? "❚❚" : "▶"}</button>
        <span>{formatTime(currentTime)}</span>
        <input type="range" min="0" max={duration || 0} step="0.01" value={Math.min(currentTime, duration || 0)} onChange={(event) => seek(Number(event.target.value))} aria-label="Seek both videos" />
        <span>{formatTime(duration)}</span>
        <strong>Synced playback</strong>
      </div>

      <div className="video-thumbnail-track" aria-label="Relative depth videos">
        {samples.map((item, itemIndex) => (
          <button
            type="button"
            key={item.id}
            ref={(node) => { thumbnails.current[itemIndex] = node; }}
            className={index === itemIndex ? "active" : ""}
            onClick={() => setIndex(itemIndex)}
            aria-pressed={index === itemIndex}
          >
            <img src={posterAsset(item.id)} alt="" />
            <span><i>{String(itemIndex + 1).padStart(2, "0")}</i>{item.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
