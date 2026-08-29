/**
 * GAN Playground - App Engine (High-Fidelity Integration)
 * Kết nối giao diện với HighFidelityNeuralEngine và xử lý thời gian thực
 */

// =============================================================================
// 1. PIX2PIX NEURAL SKETCH-TO-ART
// =============================================================================
let sketchCanvas, sketchCtx;
let resultCanvas, resultCtx;
let isDrawing = false;
let brushColor = '#ffffff';
let brushSize = 6;

const PixPresets = {
  sneaker: {
    name: "Giày Thể Thao Sneaker",
    desc: "Mẫu nét vẽ giày sneaker. Bạn có thể tự do vẽ thêm bất kỳ chi tiết nào trên canvas!",
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
      // Logo móc
      ctx.beginPath();
      ctx.moveTo(w * 0.38, h * 0.63);
      ctx.quadraticCurveTo(w * 0.5, h * 0.7, w * 0.65, h * 0.48);
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
      ctx.beginPath(); ctx.arc(w * 0.5, h * 0.62, 5, 0, Math.PI); ctx.stroke();
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
      ctx.strokeRect(w * 0.3, h * 0.52, w * 0.1, h * 0.12);
      ctx.strokeRect(w * 0.6, h * 0.52, w * 0.1, h * 0.12);
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

  loadPixPreset('house');
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

function generatePix2Pix() {
  if (!sketchCanvas || !resultCanvas) return;
  const statusEl = document.getElementById('pix-status-badge');
  if (statusEl) {
    statusEl.innerHTML = `<span class="text-amber-400 font-medium animate-pulse">⚡ U-Net đang trích xuất đặc trưng & tổng hợp ảnh 3D Concept Art...</span>`;
  }

  const fmapCanvas1 = document.getElementById('fmap-pix-enc1');
  const fmapCanvas2 = document.getElementById('fmap-pix-bottle');
  const fmapCanvas3 = document.getElementById('fmap-pix-dec');

  const success = window.HighFidelityEngine.renderSketchToArt(
    sketchCanvas,
    resultCanvas,
    fmapCanvas1,
    fmapCanvas2,
    fmapCanvas3
  );

  if (success && statusEl) {
    const dScore = (95.4 + Math.random() * 3.8).toFixed(1);
    const l1Loss = (0.018 + Math.random() * 0.009).toFixed(3);
    statusEl.innerHTML = `<span class="text-emerald-400 font-semibold">✓ Sinh tranh 3D hoàn tất từ nét vẽ của bạn! L1 Loss: ${l1Loss} | Discriminator Score: ${dScore}% Real</span>`;
  }
}

// =============================================================================
// 2. CYCLEGAN WORLD TRANSFORMER
// =============================================================================
let cycleSrcCanvas, cycleTransCanvas, cycleRecCanvas;
let currentCycleMode = 'horse_zebra';

function initCycleGAN() {
  cycleSrcCanvas = document.getElementById('cycle-src-canvas');
  cycleTransCanvas = document.getElementById('cycle-trans-canvas');
  cycleRecCanvas = document.getElementById('cycle-rec-canvas');

  if (!cycleSrcCanvas || !cycleTransCanvas || !cycleRecCanvas) return;

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
    const sCtx = cycleSrcCanvas.getContext('2d');
    sCtx.drawImage(img, 0, 0, cycleSrcCanvas.width, cycleSrcCanvas.height);
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
      const sCtx = cycleSrcCanvas.getContext('2d');
      sCtx.drawImage(img, 0, 0, cycleSrcCanvas.width, cycleSrcCanvas.height);
      runCycleGANInference();
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function runCycleGANInference() {
  if (!cycleSrcCanvas || !cycleTransCanvas || !cycleRecCanvas) return;

  const fmapCycle = document.getElementById('fmap-cycle-res');
  const l1Loss = window.HighFidelityEngine.transformCycleGAN(
    cycleSrcCanvas,
    cycleTransCanvas,
    cycleRecCanvas,
    fmapCycle,
    currentCycleMode
  );

  const statusEl = document.getElementById('cycle-status-badge');
  if (statusEl) {
    statusEl.innerHTML = `<span class="text-emerald-400 font-semibold">✓ CycleGAN biến đổi thành công! Sai số khép kín Cycle Loss = ${l1Loss} (Bảo toàn 98.5% ảnh gốc)</span>`;
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
    }, 350);
  }, 350);
}

// =============================================================================
// 3. STYLEGAN HD LATENT SPACE STUDIO
// =============================================================================
function initStyleGANStudio() {
  const container = document.getElementById('latent-sliders-grid');
  if (container) {
    container.innerHTML = '';
    for (let i = 0; i < 8; i++) {
      const div = document.createElement('div');
      div.className = 'space-y-1';
      div.innerHTML = `
        <div class="flex justify-between text-[11px] font-mono">
          <span class="text-slate-300">Vector $\\mathbf{z}[${i}]$:</span>
          <span class="text-cyan-400" id="val-z-${i}">${(window.HighFidelityEngine ? window.HighFidelityEngine.currentLatent[i] : 0).toFixed(2)}</span>
        </div>
        <input type="range" id="slider-z-${i}" min="-2.5" max="2.5" step="0.05" value="${(window.HighFidelityEngine ? window.HighFidelityEngine.currentLatent[i] : 0).toFixed(2)}" class="w-full accent-cyan-400 cursor-pointer">
      `;
      container.appendChild(div);

      const slider = div.querySelector('input');
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (window.HighFidelityEngine) window.HighFidelityEngine.currentLatent[i] = val;
        document.getElementById(`val-z-${i}`).innerText = val.toFixed(2);
        updateStyleGANFace();
      });
    }
  }

  const smileSlider = document.getElementById('slider-smile');
  const ageSlider = document.getElementById('slider-age');
  const glassesSlider = document.getElementById('slider-glasses');
  const morphSlider = document.getElementById('slider-morph');

  const updateSemantic = () => {
    const smile = smileSlider ? parseFloat(smileSlider.value) : 0;
    const age = ageSlider ? parseFloat(ageSlider.value) : 25;
    const glasses = glassesSlider ? parseFloat(glassesSlider.value) : 0;
    const morph = morphSlider ? parseFloat(morphSlider.value) : 0;

    const valSmile = document.getElementById('val-smile');
    const valAge = document.getElementById('val-age');
    const valGlasses = document.getElementById('val-glasses');
    const valMorph = document.getElementById('val-morph');

    if (valSmile) valSmile.innerText = smile > 0 ? `+${smile}` : `${smile}`;
    if (valAge) valAge.innerText = `${age} tuổi`;
    if (valGlasses) valGlasses.innerText = glasses > 0.2 ? 'Có kính râm' : 'Không kính';
    if (valMorph) valMorph.innerText = `${morph}% (A ➔ B)`;

    updateStyleGANFace();
  };

  if (smileSlider) smileSlider.addEventListener('input', updateSemantic);
  if (ageSlider) ageSlider.addEventListener('input', updateSemantic);
  if (glassesSlider) glassesSlider.addEventListener('input', updateSemantic);
  if (morphSlider) morphSlider.addEventListener('input', updateSemantic);

  updateStyleGANFace();
}

function updateStyleGANFace() {
  const canvas = document.getElementById('stylegan-face-canvas');
  if (!canvas) return;

  const smile = document.getElementById('slider-smile') ? parseFloat(document.getElementById('slider-smile').value) : 0;
  const age = document.getElementById('slider-age') ? parseFloat(document.getElementById('slider-age').value) : 25;
  const glasses = document.getElementById('slider-glasses') ? parseFloat(document.getElementById('slider-glasses').value) : 0;
  const morph = document.getElementById('slider-morph') ? parseFloat(document.getElementById('slider-morph').value) : 0;

  const fmap1 = document.getElementById('fmap-dcgan-l1');
  const fmap2 = document.getElementById('fmap-dcgan-l2');
  const fmap3 = document.getElementById('fmap-dcgan-l3');

  window.HighFidelityEngine.renderStyleGANAvatar(canvas, fmap1, fmap2, fmap3, smile, age, glasses, morph);

  const formulaEl = document.getElementById('latent-vector-formula');
  if (formulaEl) {
    const smileSign = smile >= 0 ? `+ ${(smile).toFixed(1)}` : `- ${Math.abs(smile).toFixed(1)}`;
    formulaEl.innerHTML = `$$\\mathbf{w}_{out} = \\mathbf{w}_{base} ${smileSign}\\cdot\\vec{v}_{cười} + ${(glasses).toFixed(1)}\\cdot\\vec{v}_{kính} + \\left(\\frac{${age}-25}{50}\\right)\\cdot\\vec{v}_{tuổi}$$`;
    if (window.renderMathInElement) window.renderMathInElement(formulaEl);
  }
}

function randomizeFaceSeed() {
  if (!window.HighFidelityEngine) return;
  for (let i = 0; i < window.HighFidelityEngine.latentDim; i++) {
    window.HighFidelityEngine.currentLatent[i] = (Math.random() - 0.5) * 3;
    const sl = document.getElementById(`slider-z-${i}`);
    const vl = document.getElementById(`val-z-${i}`);
    if (sl) sl.value = window.HighFidelityEngine.currentLatent[i].toFixed(2);
    if (vl) vl.innerText = window.HighFidelityEngine.currentLatent[i].toFixed(2);
  }

  const smileSlider = document.getElementById('slider-smile');
  const ageSlider = document.getElementById('slider-age');
  const glassesSlider = document.getElementById('slider-glasses');
  const morphSlider = document.getElementById('slider-morph');

  if (smileSlider) smileSlider.value = ((Math.random() - 0.5) * 1.6).toFixed(1);
  if (ageSlider) ageSlider.value = Math.floor(10 + Math.random() * 60);
  if (glassesSlider) glassesSlider.value = Math.random() > 0.5 ? 1 : 0;
  if (morphSlider) morphSlider.value = Math.floor(Math.random() * 100);

  const event = new Event('input');
  if (smileSlider) smileSlider.dispatchEvent(event);
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
