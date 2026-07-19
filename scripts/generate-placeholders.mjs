import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const baseDir = new URL("../public", import.meta.url);
const images = [
  {
    name: "robotics-baseline.svg",
    title: "Robotics · Baseline",
    primary: "#cfd6e2",
    secondary: "#c3ccd9",
    text: "#6b7aa1"
  },
  {
    name: "robotics-ours.svg",
    title: "Robotics · Ours",
    primary: "#8fb0ff",
    secondary: "#6a98ff",
    text: "#2c4ab5"
  },
  {
    name: "generation-baseline.svg",
    title: "Image Generation · Baseline",
    primary: "#cfd6e2",
    secondary: "#c3ccd9",
    text: "#6b7aa1"
  },
  {
    name: "generation-ours.svg",
    title: "Image Generation · Ours",
    primary: "#8fb0ff",
    secondary: "#6a98ff",
    text: "#2c4ab5"
  },
  {
    name: "driving-baseline.svg",
    title: "Autonomous Driving · Baseline",
    primary: "#cfd6e2",
    secondary: "#c3ccd9",
    text: "#6b7aa1"
  },
  {
    name: "driving-ours.svg",
    title: "Autonomous Driving · Ours",
    primary: "#8fb0ff",
    secondary: "#6a98ff",
    text: "#2c4ab5"
  },
  {
    name: "video-baseline.svg",
    title: "Video Synthesis · Baseline",
    primary: "#cfd6e2",
    secondary: "#c3ccd9",
    text: "#6b7aa1"
  },
  {
    name: "video-ours.svg",
    title: "Video Synthesis · Ours",
    primary: "#8fb0ff",
    secondary: "#6a98ff",
    text: "#2c4ab5"
  }
];

const template = ({ title, primary, secondary, text }) => `<?xml version="1.0" encoding="UTF-8"?>
<svg width="1400" height="800" viewBox="0 0 1400 800" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#eef4ff"/>
      <stop offset="100%" stop-color="#d9e4ff"/>
    </linearGradient>
  </defs>
  <rect width="1400" height="800" rx="36" fill="url(#bg)"/>
  <rect x="160" y="200" width="520" height="360" rx="28" fill="${primary}"/>
  <rect x="720" y="240" width="520" height="320" rx="28" fill="${secondary}"/>
  <text x="120" y="640" fill="${text}" font-size="48" font-family="Inter, sans-serif">${title}</text>
</svg>
`;

await mkdir(new URL("./images", baseDir), { recursive: true });
await mkdir(new URL("./models", baseDir), { recursive: true });

for (const image of images) {
  const filePath = new URL(`./images/${image.name}`, baseDir);
  await writeFile(filePath, template(image), "utf8");
}

await writeFile(new URL("./models/placeholder.glb", baseDir), "REPLACE_WITH_GLB", "utf8");
console.log("Placeholder assets generated.");
