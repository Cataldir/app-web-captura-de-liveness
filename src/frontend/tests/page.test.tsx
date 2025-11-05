import { render, screen, waitFor } from "@testing-library/react";

import HomePage from "../app/page";

class FakeMediaStream {}

class FakeMediaRecorder {
  public ondataavailable: ((event: { data: { size: number; arrayBuffer: () => Promise<ArrayBuffer> } }) => void) | null = null;

  constructor(public readonly stream: FakeMediaStream, _config?: MediaRecorderOptions) {}

  start(): void {
    this.ondataavailable?.({
      data: {
        size: 0,
        arrayBuffer: async () => new ArrayBuffer(0)
      }
    });
  }

  stop(): void {}
}

class FakeWebSocket {
  static OPEN = 1;
  public readyState = FakeWebSocket.OPEN;
  public onopen: (() => void) | null = null;
  public onmessage: ((event: MessageEvent<string>) => void) | null = null;
  public onclose: (() => void) | null = null;
  public onerror: (() => void) | null = null;
  public binaryType: BinaryType = "blob";

  constructor(_url: string) {
    setTimeout(() => this.onopen?.(), 0);
  }

  send(): void {}

  close(): void {
    this.onclose?.();
  }
}

describe("HomePage", () => {
  beforeAll(() => {
    Object.defineProperty(globalThis, "MediaRecorder", {
      configurable: true,
      writable: true,
      value: FakeMediaRecorder
    });

    Object.defineProperty(globalThis, "WebSocket", {
      configurable: true,
      writable: true,
      value: FakeWebSocket
    });

    Object.defineProperty(globalThis.navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: () => Promise.resolve(new FakeMediaStream())
      }
    });

    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: () => Promise.resolve()
    });
  });

  it("renderiza o título e o status inicial", async () => {
    render(<HomePage />);

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /captura de liveness/i })).toBeInTheDocument();
    });

    expect(screen.getByText(/conectando ao backend/i)).toBeInTheDocument();
  });
});
