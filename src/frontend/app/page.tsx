"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type BackendLivenessMessage, parseBackendMessage } from "../lib/liveness-message";

const WS_URL = process.env.NEXT_PUBLIC_BACKEND_WS_URL ?? "ws://localhost:8000/ws/liveness";

export default function HomePage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const [error, setError] = useState<string | null>(null);
  const [livenessResult, setLivenessResult] = useState<BackendLivenessMessage | null>(null);

  const connectWebSocket = useCallback(
    (stream: MediaStream) => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        return;
      }

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      setConnectionStatus("connecting");

      ws.onopen = () => {
        setConnectionStatus("connected");
        try {
          const recorder = new MediaRecorder(stream, {
            mimeType: "video/webm;codecs=vp8",
            videoBitsPerSecond: 250_000
          });

          recorder.ondataavailable = async (event) => {
            if (event.data.size === 0 || ws.readyState !== WebSocket.OPEN) {
              return;
            }
            try {
              const buffer = await event.data.arrayBuffer();
              ws.send(buffer);
            } catch (sendError) {
              console.error("Failed to send frame", sendError);
            }
          };

          recorder.start(750);
          mediaRecorderRef.current = recorder;
        } catch (recorderError) {
          console.error("MediaRecorder não disponível", recorderError);
          setError("Navegador não suporta MediaRecorder");
          ws.close();
        }
      };

      ws.onmessage = (event) => {
        try {
          const message = parseBackendMessage(event.data);
          setLivenessResult(message);
        } catch (parseError) {
          console.error("Mensagem inválida recebida", parseError);
        }
      };

      ws.onerror = () => {
        setError("Erro na conexão com o backend");
        setConnectionStatus("disconnected");
      };

      ws.onclose = () => {
        mediaRecorderRef.current?.stop();
        mediaRecorderRef.current = null;
        setConnectionStatus("disconnected");
        socketRef.current = null;
      };

      socketRef.current = ws;
    },
    []
  );

  useEffect(() => {
    let activeStream: MediaStream;

    const boot = async () => {
      if (typeof window === "undefined" || !navigator?.mediaDevices) {
        setError("API de mídia não disponível");
        setConnectionStatus("disconnected");
        return;
      }
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = activeStream;
          await videoRef.current.play();
        }
        connectWebSocket(activeStream);
      } catch (mediaError) {
        setError("Não foi possível acessar a câmera");
        setConnectionStatus("disconnected");
        console.error(mediaError);
      }
    };

    void boot();

    return () => {
      activeStream?.getTracks().forEach((track) => track.stop());
      mediaRecorderRef.current?.stop();
      socketRef.current?.close(1000, "client shutdown");
      mediaRecorderRef.current = null;
      socketRef.current = null;
    };
  }, [connectWebSocket]);

  const statusBadge = useMemo(() => {
    if (connectionStatus === "connecting") {
      return "Conectando ao backend...";
    }
    if (connectionStatus === "disconnected") {
      return "Conexão encerrada";
    }
    if (!livenessResult) {
      return "Aguardando dados de liveness";
    }
    return livenessResult.isLive ? "Usuário parece vivo" : "Possível spoofing";
  }, [connectionStatus, livenessResult]);

  return (
    <main className="flex flex-col items-center gap-6 p-8">
      <header className="w-full max-w-3xl text-center">
        <h1 className="text-3xl font-bold">Captura de Liveness</h1>
        <p className="text-slate-300 mt-2">A câmera será utilizada para capturar o stream e validar se você está presente.</p>
      </header>

      <section className="w-full max-w-3xl grid gap-4 md:grid-cols-[2fr,1fr]">
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4 shadow-lg">
          <video ref={videoRef} className="w-full rounded-md border border-slate-700 bg-black" muted autoPlay playsInline />
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4 flex flex-col gap-4">
          <div className="space-y-2">
            <h2 className="text-xl font-semibold">Status</h2>
            <p className="text-sm text-slate-300">{statusBadge}</p>
          </div>
          {livenessResult && (
            <div className="space-y-1 text-sm">
              <p>
                <span className="font-semibold">Confidence:</span> {(livenessResult.confidence * 100).toFixed(1)}%
              </p>
              <p>
                <span className="font-semibold">Timestamp:</span> {new Date(livenessResult.timestamp).toLocaleTimeString()}
              </p>
              <p className="text-slate-300">{livenessResult.reason}</p>
            </div>
          )}
          {error && <p className="text-danger text-sm">{error}</p>}
        </div>
      </section>
    </main>
  );
}
