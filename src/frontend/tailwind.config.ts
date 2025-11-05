import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        success: "#16a34a",
        danger: "#dc2626",
        warning: "#f59e0b"
      }
    }
  },
  plugins: []
};

export default config;
