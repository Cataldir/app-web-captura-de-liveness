import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Liveness Capture",
  description: "Real-time liveness capture demo"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="bg-slate-900 text-white min-h-screen">{children}</body>
    </html>
  );
}
