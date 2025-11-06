export type BackendLivenessMessage = {
  isLive: boolean;
  confidence: number;
  reason: string;
  timestamp: string;
};

export function parseBackendMessage(raw: unknown): BackendLivenessMessage {
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return normalize(parsed);
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error("Mensagem inválida recebida do backend");
      }
      throw error;
    }
  }

  if (raw instanceof ArrayBuffer) {
    const decoded = new TextDecoder().decode(raw);
    return parseBackendMessage(decoded);
  }

  if (typeof raw === "object" && raw !== null) {
    return normalize(raw as Record<string, unknown>);
  }

  throw new Error("Formato inesperado recebido do backend");
}

function normalize(data: Record<string, unknown>): BackendLivenessMessage {
  const isLive = Boolean(data.is_live ?? data.isLive);
  const confidence = Number(data.confidence ?? 0);
  const reason = String(data.reason ?? "Razão desconhecida");
  const timestamp = String(data.timestamp ?? new Date().toISOString());

  if (Number.isNaN(confidence)) {
    throw new Error("Confiança inválida enviada pelo backend");
  }

  const boundedConfidence = Math.max(0, Math.min(1, confidence));

  return { isLive, confidence: boundedConfidence, reason, timestamp };
}
