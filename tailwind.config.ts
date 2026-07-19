import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#eef5ff",
          100: "#dbe8ff",
          200: "#bcd5ff",
          300: "#93b8ff",
          400: "#6a98ff",
          500: "#4a7dff",
          600: "#3561e6",
          700: "#2c4ab5",
          800: "#243a8a",
          900: "#1d2e6b"
        },
        ink: {
          50: "#f7f8fb",
          100: "#eef1f6",
          200: "#d8dee9",
          300: "#b8c1d4",
          400: "#8c99b8",
          500: "#6b7aa1",
          600: "#536086",
          700: "#414d6b",
          800: "#333c53",
          900: "#252c3f"
        }
      },
      boxShadow: {
        soft: "0 20px 45px -30px rgba(30, 58, 138, 0.35)",
        lift: "0 16px 40px -20px rgba(37, 70, 140, 0.45)",
        card: "0 12px 32px -20px rgba(15, 23, 42, 0.28)"
      },
      backgroundImage: {
        "hero-gradient": "radial-gradient(circle at top, #eaf2ff, transparent 55%), linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%)",
        "section-gradient": "linear-gradient(180deg, rgba(235,242,255,0.7), rgba(248,250,255,0.9))"
      }
    }
  },
  plugins: []
};

export default config;
