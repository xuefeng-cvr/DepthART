/** @type {import('next').NextConfig} */
const repo = process.env.NEXT_PUBLIC_REPO_NAME ?? "DepthART";
const basePath = repo ? `/${repo}` : "";
const assetPrefix = basePath ? `${basePath}/` : "";

const nextConfig = {
  output: "export",
  reactStrictMode: true,
  basePath,
  assetPrefix,
  trailingSlash: true,
  images: {
    unoptimized: true
  }
};

module.exports = nextConfig;
