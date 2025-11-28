"use client";

import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type BackendLivenessMessage, parseBackendMessage } from "../lib/liveness-message";

const ENV = (typeof globalThis !== "undefined"
  ? (globalThis as typeof globalThis & {
      process?: { env?: Record<string, string | undefined> };
    }).process?.env ?? {}
  : {});

const WS_URL = ENV.NEXT_PUBLIC_BACKEND_WS_URL ?? "ws://localhost:8000/ws/liveness";
const API_URL =
  (typeof globalThis !== "undefined"
    ? ((globalThis as typeof globalThis & { process?: { env?: Record<string, string | undefined> } }).process?.env?.
        NEXT_PUBLIC_BACKEND_HTTP_URL ?? undefined)
    : undefined) ?? "http://localhost:8000";

type SimilarityEvaluationResponse = {
  similarity: number;
  status: "approved" | "not approved";
  embeddings: {
    similarity: number;
    status: "approved" | "not approved";
  };
  model: {
    similarity: number;
    status: "approved" | "not approved";
    same_person: boolean;
    explanation: string;
  };
  face_api: {
    similarity: number;
    status: "approved" | "not approved";
    is_identical: boolean;
    confidence: number;
    reason: string;
  };
};

type StrategyTableRow = {
  key: string;
  label: string;
  samePerson: boolean;
  similarity: number;
  confidence?: number;
  note?: string;
  durationMs: number;
};

export default function HomePage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");
  const [error, setError] = useState<string | null>(null);
  const [livenessResult, setLivenessResult] = useState<BackendLivenessMessage | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [evaluationResult, setEvaluationResult] = useState<SimilarityEvaluationResponse | null>(null);
  const [evaluationDuration, setEvaluationDuration] = useState<number | null>(null);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);

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

  const handleCaptureFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setEvaluationError("Vídeo indisponível para captura. Aguarde a inicialização.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setEvaluationError("Não foi possível capturar a imagem do vídeo");
      return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/png");
    setCapturedImage(dataUrl);
    setEvaluationError(null);
  }, []);

  const handleUploadChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setUploadedImage(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setUploadedImage(reader.result);
      }
    };
    reader.onerror = () => {
      setEvaluationError("Falha ao ler a imagem selecionada");
    };
    reader.readAsDataURL(file);
  }, []);

  const clearEvaluation = useCallback(() => {
    setEvaluationResult(null);
    setEvaluationDuration(null);
    setEvaluationError(null);
  }, []);

  const evaluateImages = useCallback(async () => {
    if (!capturedImage || !uploadedImage) {
      return;
    }

    try {
      setIsEvaluating(true);
      setEvaluationError(null);

      const base64Captured = capturedImage.split(",")[1];
      const base64Uploaded = uploadedImage.split(",")[1];

      if (!base64Captured || !base64Uploaded) {
        setEvaluationError("Imagens inválidas para avaliação");
        setIsEvaluating(false);
        return;
      }

      const now = performance.now();
      const response = await fetch(`${API_URL.replace(/\/$/, "")}/images/similarity/base64`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          first_image: base64Captured,
          second_image: base64Uploaded
        })
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Falha ao avaliar as imagens");
      }

      const payload = (await response.json()) as SimilarityEvaluationResponse;
      const elapsed = performance.now() - now;
      setEvaluationResult(payload);
      setEvaluationDuration(elapsed);
    } catch (requestError: unknown) {
      if (requestError instanceof Error) {
        setEvaluationError(requestError.message);
      } else {
        setEvaluationError("Erro inesperado ao avaliar as imagens");
      }
      setEvaluationResult(null);
      setEvaluationDuration(null);
    } finally {
      setIsEvaluating(false);
    }
  }, [capturedImage, uploadedImage]);

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

  const tableRows = useMemo<StrategyTableRow[]>(() => {
    if (!evaluationResult) {
      return [];
    }

    const duration = evaluationDuration ?? 0;

    return [
      {
        key: "embeddings",
        label: "Embeddings",
        samePerson: evaluationResult.embeddings.status === "approved",
        similarity: evaluationResult.embeddings.similarity,
        durationMs: duration
      },
      {
        key: "model",
        label: "Modelo Generativo",
        samePerson: evaluationResult.model.same_person,
        similarity: evaluationResult.model.similarity,
        note: evaluationResult.model.explanation,
        durationMs: duration
      },
      {
        key: "face",
        label: "Face API",
        samePerson: evaluationResult.face_api.is_identical,
        similarity: evaluationResult.face_api.similarity,
        confidence: evaluationResult.face_api.confidence,
        note: evaluationResult.face_api.reason,
        durationMs: duration
      }
    ];
  }, [evaluationResult, evaluationDuration]);

  useEffect(() => {
    clearEvaluation();
  }, [capturedImage, uploadedImage, clearEvaluation]);

  return (
    <main className="flex flex-col items-center gap-6 p-8">
      <header className="w-full max-w-3xl text-center">
        <h1 className="text-3xl font-bold">Captura de Liveness</h1>
        <p className="text-slate-300 mt-2">A câmera será utilizada para capturar o stream e validar se você está presente.</p>
      </header>

      <section className="w-full max-w-3xl grid gap-4 md:grid-cols-[2fr,1fr]">
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4 shadow-lg flex flex-col gap-4">
          <video ref={videoRef} className="w-full rounded-md border border-slate-700 bg-black" muted autoPlay playsInline />
          <button
            type="button"
            onClick={handleCaptureFrame}
            className="rounded-md bg-blue-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:bg-slate-600"
            disabled={connectionStatus !== "connected"}
          >
            Capturar imagem
          </button>
          {capturedImage && (
            <div className="rounded-md border border-slate-700 bg-slate-900/60 p-2">
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Imagem capturada</h3>
              <img src={capturedImage} alt="Imagem capturada da webcam" className="w-full rounded" />
            </div>
          )}
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-4 flex flex-col gap-4 shadow-lg">
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
          <div className="space-y-2">
            <label className="text-sm font-semibold" htmlFor="upload-image">
              Selecionar imagem de comparação
            </label>
            <input
              id="upload-image"
              type="file"
              accept="image/*"
              onChange={handleUploadChange}
              className="block w-full rounded-md border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 file:mr-4 file:rounded-md file:border-0 file:bg-blue-500 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-blue-400"
            />
            {uploadedImage && (
              <div className="rounded-md border border-slate-700 bg-slate-900/60 p-2">
                <h3 className="text-sm font-semibold text-slate-200 mb-2">Imagem enviada</h3>
                <img src={uploadedImage} alt="Imagem selecionada para comparação" className="w-full rounded" />
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={evaluateImages}
            disabled={!capturedImage || !uploadedImage || isEvaluating}
            className="mt-auto rounded-md bg-emerald-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-600"
          >
            {isEvaluating ? "Avaliando imagens..." : "Avaliar similaridade"}
          </button>
          {evaluationError && <p className="text-sm text-red-400">{evaluationError}</p>}
        </div>
      </section>

      {evaluationResult && (
        <section className="w-full max-w-3xl rounded-lg border border-slate-700 bg-slate-800/60 p-4 shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Resultados da avaliação</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-700">
              <thead className="bg-slate-900/60">
                <tr>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Estratégia
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Mesma pessoa?
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Similaridade
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Confiança
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Observações
                  </th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Tempo (ms)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700">
                {tableRows.map((row) => (
                  <tr key={row.key} className="hover:bg-slate-900/40">
                    <td className="px-4 py-3 text-sm text-slate-200">{row.label}</td>
                    <td className="px-4 py-3 text-sm font-semibold text-slate-200">
                      {row.samePerson ? "Sim" : "Não"}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-200">{row.similarity.toFixed(3)}</td>
                    <td className="px-4 py-3 text-sm text-slate-200">
                      {typeof row.confidence === "number" ? row.confidence.toFixed(3) : "-"}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-300">{row.note ?? "-"}</td>
                    <td className="px-4 py-3 text-sm text-slate-200">{Math.round(row.durationMs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {typeof evaluationResult.similarity === "number" && (
            <p className="mt-4 text-sm text-slate-300">
              Similaridade agrupada: {(evaluationResult.similarity * 100).toFixed(2)}% (status geral: {evaluationResult.status === "approved" ? "aprovado" : "reprovado"})
            </p>
          )}
        </section>
      )}
    </main>
  );
}
