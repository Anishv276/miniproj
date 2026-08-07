/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pulse: {
          red: "#FF3B3B",
          glow: "rgba(255, 59, 59, 0.4)",
        },
        obsidian: "#0F172A",
        cyan: {
          spark: "#22D3EE",
        },
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backgroundImage: {
        'forensic-gradient': "radial-gradient(circle at center, #1E293B 0%, #0F172A 100%)",
      },
    },
  },
  plugins: [],
}
