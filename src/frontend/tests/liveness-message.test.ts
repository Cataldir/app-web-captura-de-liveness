import { describe, expect, it } from "vitest";
import { parseBackendMessage } from "../lib/liveness-message";

describe("parseBackendMessage", () => {
  it("normaliza payloads com snake_case", () => {
    const payload = parseBackendMessage(
      JSON.stringify({ is_live: true, confidence: 0.8, reason: "ok", timestamp: "2024-01-01T00:00:00Z" })
    );
    expect(payload).toEqual({ isLive: true, confidence: 0.8, reason: "ok", timestamp: "2024-01-01T00:00:00Z" });
  });

  it("lança erro quando confiança não é numérica", () => {
    expect(() => parseBackendMessage(JSON.stringify({ is_live: true, confidence: "nope" }))).toThrowError(
      /Confiança inválida/
    );
  });
});
