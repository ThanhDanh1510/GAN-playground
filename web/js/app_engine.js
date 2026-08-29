/**
 * GAN Playground - App Engine 2.0 (True Neural Execution Integration)
 * Kết nối giao diện với TensorFlow.js WebGL Engine thực tế
 */

// =============================================================================
// 1. PIX2PIX NEURAL SKETCH-TO-ART (CHẠY U-NET TENSORFLOW.JS THẬT)
// =============================================================================
let sketchCanvas, sketchCtx;
let resultCanvas, resultCtx;
let isDrawing = false;
let brushColor = '#ffffff';
let brushSize = 6;

const PixPresets = {
  sneaker: {
    name: "Giày Thể Thao Sneaker",
    desc: "Mẫu nét vẽ giày. Bạn có thể tự do vẽ thêm bất kỳ chi tiết nào trên canvas!",
    draw: (ctx, w, h) => {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(w * 0.15, h * 0.7);
      ctx.quadraticCurveTo(w * 0.15, h * 0.5, w * 0.3, h * 0.5);
      ctx.lineTo(w * 0.45, h * 0.5);
      ctx.lineTo(w * 0.58, h * 0.3);
      ctx.lineTo(w * 0.72, h * 0.33);
      ctx.lineTo(w * 0.76, h * 0.5);
      ctx.quadraticCurveTo(w * 0.88, h * 0.53, w * 0.9, h * 0.65);
      ctx.lineTo(w * 0.9, h * 0.75);
      ctx.lineTo(w * 0.15, h * 0.75);
      ctx.closePath();
      ctx.stroke();
      // Đế
      ctx.beginPath();
      ctx.moveTo(w * 0.12, h * 0.75);
      ctx.lineTo(w * 0.92, h * 0.75);
      ctx.quadraticCurveTo(w * 0.92, h * 0.85, w * 0.82, h * 0.85);
      ctx.lineTo(w * 0.2, h * 0.85);
      ctx.quadraticCurveTo(w * 0.12, h * 0.85, w * 0.12, h * 0.75);
      ctx.stroke();
    }
  },
  cat: {
    name: "Mèo Hoạt Hình Cute",
    desc: "Mẫu nét vẽ mặt mèo. Thử vẽ thêm mắt to, tai thỏ hoặc người que!",
    draw: (ctx, w, h) => {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.ellipse(w * 0.5, h * 0.55, w * 0.25, h * 0.28, 0, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(w * 0.32, h * 0.35); ctx.lineTo(w * 0.25, h * 0.15); ctx.lineTo(w * 0.42, h * 0.28);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(w * 0.68, h * 0.35); ctx.lineTo(w * 0.75, h * 0.15); ctx.lineTo(w * 0.58, h * 0.28);
      ctx.stroke();
      ctx.beginPath(); ctx.ellipse(w * 0.42, h * 0.5, 8, 12, 0, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(w * 0.58, h * 0.5, 8, 12, 0, 0, 2 * Math.PI); ctx.stroke();
    }
  },
  house: {
    name: "Ngôi Nhà Cổ Tích",
    desc: "Mẫu ngôi nhà. Bạn có thể vẽ thêm ống khói, mây trời hoặc cây cối!",
    draw: (ctx, w, h) => {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 5;
      ctx.strokeRect(w * 0.25, h * 0.45, w * 0.5, h * 0.45);
      ctx.beginPath();
      ctx.moveTo(w * 0.2, h * 0.45); ctx.lineTo(w * 0.5, h * 0.18); ctx.lineTo(w * 0.8, h * 0.45);
      ctx.closePath();
      ctx.stroke();
      ctx.strokeRect(w * 0.43, h * 0.62, w * 0.14, h * 0.28);
    }
  }
};

function initPix2Pix() {
  sketchCanvas = document.getElementById('pix-sketch-canvas');
  resultCanvas = document.getElementById('pix-result-canvas');
  if (!sketchCanvas || !resultCanvas) return;

  sketchCtx = sketchCanvas.getContext('2d');
  resultCtx = resultCanvas.getContext('2d');

  clearSketchPad();

  // Mouse & Touch
  const startDraw = (e) => {
    isDrawing = true;
    const pos = getCanvasPos(e);
    sketchCtx.beginPath();
    sketchCtx.moveTo(pos.x, pos.y);
  };
  const draw = (e) => {
    if (!isDrawing) return;
    const pos = getCanvasPos(e);
    sketchCtx.strokeStyle = brushColor;
    sketchCtx.lineWidth = brushSize;
    sketchCtx.lineCap = 'round';
    sketchCtx.lineJoin = 'round';
    sketchCtx.lineTo(pos.x, pos.y);
    sketchCtx.stroke();
  };
  const stopDraw = () => { isDrawing = false; };

  sketchCanvas.addEventListener('mousedown', startDraw);
  sketchCanvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDraw);

  sketchCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e.touches[0]); });
  sketchCanvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e.touches[0]); });
  sketchCanvas.addEventListener('touchend', stopDraw);

  loadPixPreset('sneaker');
}

function getCanvasPos(e) {
  const rect = sketchCanvas.getBoundingClientRect();
  const scaleX = sketchCanvas.width / rect.width;
  const scaleY = sketchCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function clearSketchPad() {
  if (!sketchCtx || !resultCtx) return;
  sketchCtx.fillStyle = '#0f172a';
  sketchCtx.fillRect(0, 0, sketchCanvas.width, sketchCanvas.height);
  
  resultCtx.fillStyle = '#0f172a';
  resultCtx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
}

function loadPixPreset(key) {
  const data = PixPresets[key];
  if (!data) return;

  const descEl = document.getElementById('pix-preset-desc');
  if (descEl) descEl.innerText = data.desc;

  clearSketchPad();
  data.draw(sketchCtx, sketchCanvas.width, sketchCanvas.height);
  generatePix2Pix();
}

/**
 * Chạy Mạng Nơ-ron U-Net Thật qua TensorFlow.js WebGL Backend
 */
async function generatePix2Pix() {
  if (!sketchCanvas || !resultCanvas) return;
  const statusEl = document.getElementById('pix-status-badge');
  if (statusEl) {
    statusEl.innerHTML = `<span class="text-amber-400 font-medium animate-pulse">⚡ Đang chuyển ảnh thành Tensor & chạy forward pass qua U-Net (WebGL GPU)...</span>`;
  }

  // 1. Chạy qua U-Net Neural Engine thật
  if (window.NeuralEngine && window.NeuralEngine.isReady) {
    const { resultImage, encFeatures, bottleneckFeatures, decFeatures } = window.NeuralEngine.processSketchWithUNet(sketchCanvas);
    
    // Render kết quả Tensor ra Canvas chính
    await tf.browser.toPixels(resultImage, resultCanvas);

    // 2. Render Kính hiển vi Nơ-ron (Feature Maps)
    const fmapCanvas1 = document.getElementById('fmap-pix-enc1');
    const fmapCanvas2 = document.getElementById('fmap-pix-bottle');
    const fmapCanvas3 = document.getElementById('fmap-pix-dec');

    if (fmapCanvas1) await window.NeuralEngine.renderFeatureMapSlice(encFeatures, 4, fmapCanvas1);
    if (fmapCanvas2) await window.NeuralEngine.renderFeatureMapSlice(bottleneckFeatures, 12, fmapCanvas2);
    if (fmapCanvas3) await window.NeuralEngine.renderFeatureMapSlice(decFeatures, 8, fmapCanvas3);

    // Dọn dẹp bộ nhớ GPU
    resultImage.dispose();
    encFeatures.dispose();
    bottleneckFeatures.dispose();
    decFeatures.dispose();

    if (statusEl) {
      statusEl.innerHTML = `<span class="text-emerald-400 font-semibold">✓ U-Net Forward Pass thành công trên WebGL GPU! Tensor shape: [64, 64, 3] | Vui lòng xem Feature Maps bên dưới.</span>`;
    }
  }
}

// =============================================================================
// 2. CYCLEGAN DYNAMIC RESIDUAL TRANSFORMER
// =============================================================================
let cycleSrcCanvas, cycleSrcCtx;
let cycleTransCanvas, cycleTransCtx;
let cycleRecCanvas, cycleRecCtx;
let currentCycleMode = 'horse_zebra';

function initCycleGAN() {
  cycleSrcCanvas = document.getElementById('cycle-src-canvas');
  cycleTransCanvas = document.getElementById('cycle-trans-canvas');
  cycleRecCanvas = document.getElementById('cycle-rec-canvas');

  if (!cycleSrcCanvas || !cycleTransCanvas || !cycleRecCanvas) return;

  cycleSrcCtx = cycleSrcCanvas.getContext('2d');
  cycleTransCtx = cycleTransCanvas.getContext('2d');
  cycleRecCtx = cycleRecCanvas.getContext('2d');

  const modelSelect = document.getElementById('cyclegan-model-select');
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      currentCycleMode = e.target.value;
      loadCyclePreset(currentCycleMode);
    });
  }

  loadCyclePreset('horse_zebra');
}

function loadCyclePreset(mode) {
  currentCycleMode = mode;
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    cycleSrcCtx.drawImage(img, 0, 0, cycleSrcCanvas.width, cycleSrcCanvas.height);
    runCycleGANInference();
  };

  if (mode === 'horse_zebra') img.src = 'assets/horse1.svg';
  else if (mode === 'summer_winter') img.src = 'assets/summer1.svg';
  else img.src = 'assets/village.svg';
}

function handleCycleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      cycleSrcCtx.drawImage(img, 0, 0, cycleSrcCanvas.width, cycleSrcCanvas.height);
      runCycleGANInference();
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

async function runCycleGANInference() {
  if (!cycleSrcCanvas || !cycleTransCanvas || !cycleRecCanvas) return;

  if (window.NeuralEngine && window.NeuralEngine.isReady) {
    const { fakeB, reconA, cycleLoss, features } = window.NeuralEngine.processCycleGAN(cycleSrcCanvas);

    await tf.browser.toPixels(fakeB, cycleTransCanvas);
    await tf.browser.toPixels(reconA, cycleRecCanvas);

    // Kính hiển vi cho CycleGAN Residual Block
    const fmapCycle = document.getElementById('fmap-cycle-res');
    if (fmapCycle) await window.NeuralEngine.renderFeatureMapSlice(features, 8, fmapCycle);

    fakeB.dispose();
    reconA.dispose();
    features.dispose();

    const statusEl = document.getElementById('cycle-status-badge');
    if (statusEl) {
      statusEl.innerHTML = `<span class="text-emerald-400 font-semibold">✓ CycleGAN Forward & Inverse Pass hoàn tất trên GPU! Sai số ma trận L1 Loss: ${(cycleLoss).toFixed(4)}</span>`;
    }
  }
}

function triggerCycleAnimation() {
  const transCard = document.getElementById('cycle-trans-card');
  const recCard = document.getElementById('cycle-rec-card');
  if (transCard) transCard.classList.add('ring-4', 'ring-cyan-400');
  setTimeout(() => {
    if (transCard) transCard.classList.remove('ring-4', 'ring-cyan-400');
    if (recCard) recCard.classList.add('ring-4', 'ring-emerald-400');
    setTimeout(() => {
      if (recCard) recCard.classList.remove('ring-4', 'ring-emerald-400');
      runCycleGANInference();
    }, 400);
  }, 400);
}

// =============================================================================
// 3. STYLEGAN / DCGAN 100-D LATENT SPACE SYNTHESIZER
// =============================================================================
let latentSliders = [];

function initStyleGANStudio() {
  const container = document.getElementById('latent-sliders-grid');
  if (container) {
    container.innerHTML = '';
    // Tạo 8 thanh trượt điều khiển trực tiếp các chiều chính z[0] ... z[7] của tensor
    for (let i = 0; i < 8; i++) {
      const div = document.createElement('div');
      div.className = 'space-y-1';
      div.innerHTML = `
        <div class="flex justify-between text-[11px] font-mono">
          <span class="text-slate-300">Chiều Tensor $\\mathbf{z}[${i}]$:</span>
          <span class="text-cyan-400" id="val-z-${i}">0.00</span>
        </div>
        <input type="range" id="slider-z-${i}" min="-2.5" max="2.5" step="0.05" value="${(window.NeuralEngine ? window.NeuralEngine.currentLatent[i] : 0).toFixed(2)}" class="w-full accent-cyan-400 cursor-pointer">
      `;
      container.appendChild(div);

      const slider = div.querySelector('input');
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (window.NeuralEngine) window.NeuralEngine.currentLatent[i] = val;
        document.getElementById(`val-z-${i}`).innerText = val.toFixed(2);
        updateDCGANFace();
      });
    }
  }

  // Sliders ngữ nghĩa (Nụ cười, Tuổi, Kính)
  const smileSlider = document.getElementById('slider-smile');
  const ageSlider = document.getElementById('slider-age');
  const glassesSlider = document.getElementById('slider-glasses');

  const updateSemantic = () => {
    if (!window.NeuralEngine) return;
    const smile = smileSlider ? parseFloat(smileSlider.value) : 0;
    const age = ageSlider ? parseFloat(ageSlider.value) : 25;
    const glasses = glassesSlider ? parseFloat(glassesSlider.value) : 0;

    // Phép cộng trừ vector tiềm ẩn trên Tensor thật
    window.NeuralEngine.currentLatent[0] = smile * 1.5;
    window.NeuralEngine.currentLatent[1] = ((age - 25) / 50) * 1.8;
    window.NeuralEngine.currentLatent[2] = glasses * 2.0;

    // Cập nhật giá trị lên các slider z[0..2]
    for (let k = 0; k < 3; k++) {
      const sl = document.getElementById(`slider-z-${k}`);
      const vl = document.getElementById(`val-z-${k}`);
      if (sl) sl.value = window.NeuralEngine.currentLatent[k].toFixed(2);
      if (vl) vl.innerText = window.NeuralEngine.currentLatent[k].toFixed(2);
    }

    updateDCGANFace();
  };

  if (smileSlider) smileSlider.addEventListener('input', updateSemantic);
  if (ageSlider) ageSlider.addEventListener('input', updateSemantic);
  if (glassesSlider) glassesSlider.addEventListener('input', updateSemantic);

  updateDCGANFace();
}

async function updateDCGANFace() {
  const canvas = document.getElementById('stylegan-face-canvas');
  if (!canvas || !window.NeuralEngine || !window.NeuralEngine.isReady) return;

  const { image, fMap1, fMap2, fMap3 } = window.NeuralEngine.generateDCGAN(window.NeuralEngine.currentLatent);
  
  // Render ảnh RGB tensor ra Canvas
  await tf.browser.toPixels(image, canvas);

  // Kính hiển vi soi các tầng Deconv
  const fmap1Canvas = document.getElementById('fmap-dcgan-l1');
  const fmap2Canvas = document.getElementById('fmap-dcgan-l2');
  const fmap3Canvas = document.getElementById('fmap-dcgan-l3');

  if (fmap1Canvas) await window.NeuralEngine.renderFeatureMapSlice(fMap1, 4, fmap1Canvas);
  if (fmap2Canvas) await window.NeuralEngine.renderFeatureMapSlice(fMap2, 8, fmap2Canvas);
  if (fmap3Canvas) await window.NeuralEngine.renderFeatureMapSlice(fMap3, 2, fmap3Canvas);

  image.dispose();
  fMap1.dispose();
  fMap2.dispose();
  fMap3.dispose();

  // Cập nhật công thức
  const formulaEl = document.getElementById('latent-vector-formula');
  if (formulaEl) {
    formulaEl.innerHTML = `$$\\mathbf{z} = [${window.NeuralEngine.currentLatent[0].toFixed(2)}, ${window.NeuralEngine.currentLatent[1].toFixed(2)}, ${window.NeuralEngine.currentLatent[2].toFixed(2)}, \\dots, ${window.NeuralEngine.currentLatent[7].toFixed(2)}] \\in \\mathbb{R}^{32}$$`;
    if (window.renderMathInElement) window.renderMathInElement(formulaEl);
  }
}

function randomizeFaceSeed() {
  if (!window.NeuralEngine) return;
  for (let i = 0; i < window.NeuralEngine.latentDim; i++) {
    window.NeuralEngine.currentLatent[i] = (Math.random() - 0.5) * 3;
    const sl = document.getElementById(`slider-z-${i}`);
    const vl = document.getElementById(`val-z-${i}`);
    if (sl) sl.value = window.NeuralEngine.currentLatent[i].toFixed(2);
    if (vl) vl.innerText = window.NeuralEngine.currentLatent[i].toFixed(2);
  }
  updateDCGANFace();
}

// =============================================================================
// 4. MINI-GAME THÁM TỬ AI
// =============================================================================
const TuringQuestions = [
  {
    title: "Vòng 1: Đôi mắt và Bông tai",
    desc: "Hãy chú ý độ đối xứng của hai bên bông tai và con ngươi trong mắt!",
    imgReal: "assets/turing_real_1.svg",
    imgFake: "assets/turing_fake_1.svg",
    fakeSide: "B",
    explanation: "Ảnh B là ảnh do AI tạo ra (Fake)! AI thường vẽ hai bên bông tai không giống nhau và viền mắt hơi bị nhòe bất đối xứng."
  },
  {
    title: "Vòng 2: Hậu cảnh & Nếp tóc",
    desc: "Hãy chú ý các sợi tóc hòa vào phông nền phía sau!",
    imgReal: "assets/turing_real_2.svg",
    imgFake: "assets/turing_fake_2.svg",
    fakeSide: "A",
    explanation: "Ảnh A là ảnh do AI tạo ra! Các sợi tóc hòa tan bất thường vào họa tiết bức tường phía sau - đây là lỗi texture kinh điển của GAN."
  },
  {
    title: "Vòng 3: Bàn tay và Ngón tay",
    desc: "AI tạo sinh thời kỳ đầu rất sợ vẽ bàn tay!",
    imgReal: "assets/turing_real_3.svg",
    imgFake: "assets/turing_fake_3.svg",
    fakeSide: "B",
    explanation: "Ảnh B là ảnh do AI tạo ra! Bàn tay có 6 ngón và các đốt ngón tay bị dính vào nhau."
  }
];

let currentQuestionIdx = 0;
let userScore = 0;

function initTuringGame() {
  loadTuringQuestion(0);
}

function loadTuringQuestion(idx) {
  if (idx >= TuringQuestions.length) {
    showTuringEndScreen();
    return;
  }
  currentQuestionIdx = idx;
  const q = TuringQuestions[idx];

  const titleEl = document.getElementById('turing-round-title');
  const descEl = document.getElementById('turing-round-desc');
  const imgA = document.getElementById('turing-img-a');
  const imgB = document.getElementById('turing-img-b');
  const expBox = document.getElementById('turing-explanation-box');

  if (titleEl) titleEl.innerText = `${q.title} (${idx + 1}/${TuringQuestions.length})`;
  if (descEl) descEl.innerText = q.desc;
  if (expBox) expBox.classList.add('hidden');

  if (q.fakeSide === 'A') {
    if (imgA) imgA.src = q.imgFake;
    if (imgB) imgB.src = q.imgReal;
  } else {
    if (imgA) imgA.src = q.imgReal;
    if (imgB) imgB.src = q.imgFake;
  }
}

function submitTuringAnswer(chosenSide) {
  const q = TuringQuestions[currentQuestionIdx];
  const isCorrect = chosenSide === q.fakeSide;
  if (isCorrect) userScore++;

  const expBox = document.getElementById('turing-explanation-box');
  const expText = document.getElementById('turing-explanation-text');
  const scoreBadge = document.getElementById('turing-score-badge');

  if (scoreBadge) scoreBadge.innerText = `Điểm thám tử: ${userScore}/${currentQuestionIdx + 1}`;
  if (expBox && expText) {
    expBox.classList.remove('hidden');
    expBox.className = `mt-4 p-4 rounded-xl border ${isCorrect ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300' : 'bg-rose-950/40 border-rose-500/50 text-rose-300'}`;
    expText.innerHTML = `<strong>${isCorrect ? '🎉 CHÍNH XÁC!' : '❌ CHƯA ĐÚNG!'}</strong> ${q.explanation}`;
  }
}

function nextTuringQuestion() {
  loadTuringQuestion(currentQuestionIdx + 1);
}

function showTuringEndScreen() {
  const container = document.getElementById('turing-game-container');
  if (container) {
    container.innerHTML = `
      <div class="text-center py-10 space-y-4">
        <div class="text-5xl">🏆</div>
        <h3 class="text-2xl font-bold text-cyan-400">Hoàn Thành Thử Thách Thám Tử AI!</h3>
        <p class="text-slate-300 text-lg">Bạn đã đạt điểm số: <strong class="text-amber-400">${userScore}/${TuringQuestions.length}</strong></p>
        <button onclick="userScore=0; loadTuringQuestion(0);" class="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold rounded-xl transition shadow-lg shadow-cyan-500/20">Chơi Lại</button>
      </div>
    `;
  }
}
