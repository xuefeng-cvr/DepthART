import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DepthART | Scaling Foundation Monocular Depth to Tiny Models",
  description: "Official project page for DepthART: Scaling Foundation Monocular Depth to Tiny Models."
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
