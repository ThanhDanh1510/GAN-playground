/**
 * Visualizer JS - Trực quan hóa điểm dữ liệu, Heatmap, và Loss Battle
 */

class GANVisualizer {
  constructor(canvasId, lossCanvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.lossCanvas = document.getElementById(lossCanvasId);
    this.lossCtx = this.lossCanvas ? this.lossCanvas.getContext('2d') : null;
  }

  // Chuyển đổi tọa độ toán học [-1.2, 1.2] sang tọa độ pixel trên canvas
  toPixel(x, y) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const px = ((x + 1.2) / 2.4) * w;
    const py = ((1.2 - y) / 2.4) * h;
    return [px, py];
  }

  renderFrame(realPoints, fakePoints, gridData) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    // 1. Vẽ Heatmap của Discriminator
    if (gridData && gridData.probs) {
      const gs = gridData.gridSize;
      const cellW = w / gs;
      const cellH = h / gs;

      for (let i = 0; i < gs; i++) {
        for (let j = 0; j < gs; j++) {
          const prob = gridData.probs[i * gs + j]; // [0, 1]
          // 0 (Fake/Generator) -> Xanh lam (#3b82f6), 1 (Real) -> Đỏ cam (#ef4444)
          // Gradient màu mượt mà
          const r = Math.floor(59 + prob * (239 - 59));
          const g = Math.floor(130 + prob * (68 - 130));
          const b = Math.floor(246 + prob * (68 - 246));
          this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.22)`;
          this.ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
        }
      }
    }

    // 2. Vẽ lưới trục tọa độ
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(w / 2, 0); this.ctx.lineTo(w / 2, h);
    this.ctx.moveTo(0, h / 2); this.ctx.lineTo(w, h / 2);
    this.ctx.stroke();

    // 3. Vẽ các điểm dữ liệu thật (Real Data Points) - Màu Cam Đỏ
    if (realPoints && realPoints.data) {
      this.ctx.fillStyle = '#f97316';
      const n = realPoints.shape[0];
      for (let i = 0; i < n; i++) {
        const [px, py] = this.toPixel(realPoints.data[i * 2], realPoints.data[i * 2 + 1]);
        this.ctx.beginPath();
        this.ctx.arc(px, py, 3.2, 0, 2 * Math.PI);
        this.ctx.fill();
      }
    }

    // 4. Vẽ các điểm do Generator sinh ra (Fake Generated Points) - Màu Xanh Neon
    if (fakePoints && fakePoints.length > 0) {
      this.ctx.fillStyle = '#06b6d4';
      this.ctx.strokeStyle = '#ffffff';
      this.ctx.lineWidth = 1;
      for (let i = 0; i < fakePoints.length; i++) {
        const [px, py] = this.toPixel(fakePoints[i][0], fakePoints[i][1]);
        this.ctx.beginPath();
        this.ctx.arc(px, py, 3.8, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.stroke();
      }
    }
  }

  renderLoss(lossHistory) {
    if (!this.lossCtx) return;
    const w = this.lossCanvas.width;
    const h = this.lossCanvas.height;
    const ctx = this.lossCtx;
    ctx.clearRect(0, 0, w, h);

    const dLoss = lossHistory.d;
    const gLoss = lossHistory.g;
    if (dLoss.length < 2) return;

    // Tìm max loss để scale
    let maxVal = 2.5;
    for (let v of dLoss) if (v > maxVal) maxVal = v;
    for (let v of gLoss) if (v > maxVal) maxVal = v;

    const len = dLoss.length;
    const stepX = w / Math.max(1, len - 1);

    // Vẽ đường D Loss (Cảnh sát - Cam)
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
      const x = i * stepX;
      const y = h - (dLoss[i] / maxVal) * (h - 10) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Vẽ đường G Loss (Kẻ làm giả - Xanh Neon)
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
      const x = i * stepX;
      const y = h - (gLoss[i] / maxVal) * (h - 10) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

if (typeof module !== 'undefined') {
  module.exports = { GANVisualizer };
}
