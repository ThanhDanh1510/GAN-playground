/**
 * GAN Playground - App Engine
 * Xử lý tương tác cho Top 3 Ứng dụng: CycleGAN, Pix2Pix, StyleGAN Latent Studio và Mini-Game
 */

// ==========================================
// 1. CYCLEGAN INTERACTIVE ENGINE
// ==========================================
const CycleGANDataset = {
  horse_zebra: {
    title: "Ngựa thường ↔ Ngựa vằn (Horse ↔ Zebra)",
    desc: "CycleGAN học cách thêm hoặc xóa các sọc vằn trên thân ngựa mà không làm biến dạng phong cảnh xung quanh!",
    domainA_name: "Ngựa Thường (Horse)",
    domainB_name: "Ngựa Vằn (Zebra)",
    presets: [
      {
        name: "Ngựa trên đồng cỏ xanh",
        imgA: "assets/horse1.svg",
        fakeB: "assets/zebra1.svg",
        recA: "assets/horse1_rec.svg"
      },
      {
        name: "Hai chú ngựa bên đồi",
        imgA: "assets/horse2.svg",
        fakeB: "assets/zebra2.svg",
        recA: "assets/horse2_rec.svg"
      }
    ]
  },
  summer_winter: {
    title: "Mùa hè ↔ Mùa đông tuyết phủ (Summer ↔ Winter)",
    desc: "Chuyển đổi thời tiết: Thêm tuyết trắng trên đỉnh núi và cây thông, thay đổi ánh sáng mặt trời.",
    domainA_name: "Mùa Hè Nắng Ấm",
    domainB_name: "Mùa Đông Tuyết Trắng",
    presets: [
      {
        name: "Dãy núi Yosemite",
        imgA: "assets/summer1.svg",
        fakeB: "assets/winter1.svg",
        recA: "assets/summer1_rec.svg"
      }
    ]
  },
  photo_vangogh: {
    title: "Ảnh chụp ↔ Tranh sơn dầu Van Gogh",
    desc: "CycleGAN học nét cọ xoáy đặc trưng và bảng màu của kiệt tác 'Đêm đầy sao' (Starry Night).",
    domainA_name: "Ảnh Chụp Thực Tế",
    domainB_name: "Phong Cách Van Gogh",
    presets: [
      {
        name: "Làng quê ban đêm",
        imgA: "assets/village.svg",
        fakeB: "assets/vangogh_village.svg",
        recA: "assets/village_rec.svg"
      }
    ]
  }
};

let currentCycleModel = 'horse_zebra';
let cycleDirection = 'AtoB'; // 'AtoB' or 'BtoA'

function initCycleGAN() {
  const modelSelect = document.getElementById('cyclegan-model-select');
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      currentCycleModel = e.target.value;
      updateCycleGANDemo();
    });
  }
  updateCycleGANDemo();
}

function updateCycleGANDemo() {
  const data = CycleGANDataset[currentCycleModel];
  if (!data) return;

  const titleEl = document.getElementById('cyclegan-title');
  const descEl = document.getElementById('cyclegan-desc');
  if (titleEl) titleEl.innerText = data.title;
  if (descEl) descEl.innerText = data.desc;

  const preset = data.presets[0];
  const srcImg = document.getElementById('cycle-src-img');
  const transImg = document.getElementById('cycle-trans-img');
  const recImg = document.getElementById('cycle-rec-img');
  const lossVal = document.getElementById('cycle-loss-val');

  if (srcImg) srcImg.src = preset.imgA;
  if (transImg) transImg.src = preset.fakeB;
  if (recImg) recImg.src = preset.recA;
  if (lossVal) lossVal.innerText = (0.012 + Math.random() * 0.008).toFixed(4);
}

function triggerCycleAnimation() {
  const transCard = document.getElementById('cycle-trans-card');
  const recCard = document.getElementById('cycle-rec-card');
  const statusEl = document.getElementById('cycle-status-badge');

  if (statusEl) {
    statusEl.innerHTML = `<span class="inline-flex items-center gap-1 text-cyan-400 animate-pulse"><svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg> Đang chạy vòng lặp Cycle GAN G(A) ➔ F(B)...</span>`;
  }

  if (transCard) transCard.classList.add('ring-2', 'ring-cyan-400');
  setTimeout(() => {
    if (transCard) transCard.classList.remove('ring-2', 'ring-cyan-400');
    if (recCard) recCard.classList.add('ring-2', 'ring-emerald-400');
    setTimeout(() => {
      if (recCard) recCard.classList.remove('ring-2', 'ring-emerald-400');
      if (statusEl) {
        statusEl.innerHTML = `<span class="text-emerald-400 font-medium">✓ Hoàn thành chu trình! Cycle Consistency Loss = ${ (0.011 + Math.random()*0.005).toFixed(4) } (Rất khớp với ảnh gốc)</span>`;
      }
    }, 600);
  }, 600);
}

// ==========================================
// 2. PIX2PIX MAGIC SKETCHPAD ENGINE
// ==========================================
let sketchCanvas, sketchCtx;
let isDrawing = false;
let brushColor = '#ffffff';
let brushSize = 4;
let currentPixPreset = 'sneaker';

const PixPresets = {
  sneaker: {
    name: "Giày Thể Thao Sneaker",
    desc: "Vẽ đường viền thân giày, đế giày và logo. U-Net Generator sẽ phủ chất liệu da, cao su và màu sắc tương phản!",
    svgEdge: "assets/sketch_sneaker.svg",
    svgColor: "assets/result_sneaker.svg"
  },
  cat: {
    name: "Mèo Hoạt Hình Cute",
    desc: "Vẽ khuôn mặt, đôi tai nhọn và râu mèo. Generator sẽ tạo bộ lông mềm mại và đôi mắt lấp lánh!",
    svgEdge: "assets/sketch_cat.svg",
    svgColor: "assets/result_cat.svg"
  },
  house: {
    name: "Ngôi Nhà Cổ Tích",
    desc: "Vẽ tường nhà, mái ngói tam giác và cửa sổ. Pix2Pix sẽ tạo gạch tường và cửa kính sáng đèn!",
    svgEdge: "assets/sketch_house.svg",
    svgColor: "assets/result_house.svg"
  }
};

function initPix2Pix() {
  sketchCanvas = document.getElementById('pix-sketch-canvas');
  if (!sketchCanvas) return;
  sketchCtx = sketchCanvas.getContext('2d');

  // Đặt nền canvas màu đen
  clearSketchPad();

  // Mouse & Touch events
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
  if (!sketchCtx) return;
  sketchCtx.fillStyle = '#0f172a';
  sketchCtx.fillRect(0, 0, sketchCanvas.width, sketchCanvas.height);
}

function loadPixPreset(presetKey) {
  currentPixPreset = presetKey;
  const data = PixPresets[presetKey];
  if (!data) return;

  const descEl = document.getElementById('pix-preset-desc');
  if (descEl) descEl.innerText = data.desc;

  const img = new Image();
  img.onload = () => {
    clearSketchPad();
    sketchCtx.drawImage(img, 0, 0, sketchCanvas.width, sketchCanvas.height);
  };
  img.src = data.svgEdge;

  // Cập nhật kết quả AI
  const resImg = document.getElementById('pix-result-img');
  if (resImg) resImg.src = data.svgColor;
}

function generatePix2Pix() {
  const resImg = document.getElementById('pix-result-img');
  const statusEl = document.getElementById('pix-status-badge');
  const data = PixPresets[currentPixPreset];

  if (statusEl) {
    statusEl.innerHTML = `<span class="text-amber-400 font-medium animate-pulse">⚡ U-Net Generator đang trích xuất đặc trưng & PatchGAN đang chấm điểm...</span>`;
  }
  if (resImg) {
    resImg.style.opacity = '0.3';
    setTimeout(() => {
      resImg.src = data ? data.svgColor : 'assets/result_sneaker.svg';
      resImg.style.opacity = '1';
      if (statusEl) {
        statusEl.innerHTML = `<span class="text-emerald-400 font-semibold">✓ Sinh ảnh thành công! L1 Loss: 0.034 | Discriminator Score: 96.8% Real</span>`;
      }
    }, 450);
  }
}

// ==========================================
// 3. STYLEGAN LATENT STUDIO ENGINE
// ==========================================
const StyleGANFaces = {
  // Biểu diễn các thành phần vector khuôn mặt
  baseFace: {
    gender: 'neutral',
    smile: 0,
    age: 25,
    glasses: 0,
    hairColor: 'brown'
  }
};

function initStyleGANStudio() {
  const smileSlider = document.getElementById('slider-smile');
  const ageSlider = document.getElementById('slider-age');
  const glassesSlider = document.getElementById('slider-glasses');
  const morphSlider = document.getElementById('slider-morph');

  const updateFace = () => {
    const smile = smileSlider ? parseFloat(smileSlider.value) : 0;
    const age = ageSlider ? parseFloat(ageSlider.value) : 25;
    const glasses = glassesSlider ? parseFloat(glassesSlider.value) : 0;
    const morph = morphSlider ? parseFloat(morphSlider.value) : 0;

    renderStyleGANCanvas(smile, age, glasses, morph);
  };

  if (smileSlider) smileSlider.addEventListener('input', updateFace);
  if (ageSlider) ageSlider.addEventListener('input', updateFace);
  if (glassesSlider) glassesSlider.addEventListener('input', updateFace);
  if (morphSlider) morphSlider.addEventListener('input', updateFace);

  updateFace();
}

function renderStyleGANCanvas(smile, age, glasses, morph) {
  const canvas = document.getElementById('stylegan-face-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Background
  const bgGrad = ctx.createRadialGradient(w/2, h/2, 20, w/2, h/2, w/2);
  bgGrad.addColorStop(0, '#1e293b');
  bgGrad.addColorStop(1, '#0f172a');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, w, h);

  // Màu da & hình dạng khuôn mặt (Morphing)
  const skinTone = morph > 50 ? '#f6d8b8' : '#fed7aa';
  ctx.fillStyle = skinTone;
  ctx.beginPath();
  const faceW = 85 - (age > 50 ? 5 : 0);
  const faceH = 110;
  ctx.ellipse(w/2, h/2 + 5, faceW, faceH, 0, 0, 2 * Math.PI);
  ctx.fill();

  // Mái tóc
  ctx.fillStyle = morph > 50 ? '#f59e0b' : '#334155'; // Tóc vàng hoặc đen
  ctx.beginPath();
  ctx.arc(w/2, h/2 - 25, faceW + 2, Math.PI, 2 * Math.PI);
  ctx.fill();

  // Mắt
  ctx.fillStyle = '#0f172a';
  const eyeOffset = 30;
  const eyeY = h/2 - 5;
  ctx.beginPath(); ctx.arc(w/2 - eyeOffset, eyeY, 6, 0, 2 * Math.PI); ctx.fill();
  ctx.beginPath(); ctx.arc(w/2 + eyeOffset, eyeY, 6, 0, 2 * Math.PI); ctx.fill();
  // Ánh sáng trong mắt
  ctx.fillStyle = '#ffffff';
  ctx.beginPath(); ctx.arc(w/2 - eyeOffset + 2, eyeY - 2, 2.5, 0, 2 * Math.PI); ctx.fill();
  ctx.beginPath(); ctx.arc(w/2 + eyeOffset + 2, eyeY - 2, 2.5, 0, 2 * Math.PI); ctx.fill();

  // Nếp nhăn tuổi tác (Age)
  if (age > 45) {
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.6)';
    ctx.lineWidth = 1.5;
    // Trán
    ctx.beginPath(); ctx.moveTo(w/2 - 35, h/2 - 55); ctx.lineTo(w/2 + 35, h/2 - 55); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w/2 - 25, h/2 - 45); ctx.lineTo(w/2 + 25, h/2 - 45); ctx.stroke();
    // Khóe mắt
    ctx.beginPath(); ctx.moveTo(w/2 - 40, eyeY); ctx.lineTo(w/2 - 48, eyeY - 4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w/2 + 40, eyeY); ctx.lineTo(w/2 + 48, eyeY - 4); ctx.stroke();
  }

  // Nụ cười (Smile)
  ctx.strokeStyle = '#e11d48';
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  const mouthY = h/2 + 50;
  const mouthCurvature = smile * 22; // -1 to 1
  ctx.beginPath();
  ctx.moveTo(w/2 - 28, mouthY);
  ctx.quadraticCurveTo(w/2, mouthY + mouthCurvature, w/2 + 28, mouthY);
  ctx.stroke();

  // Kính râm (Glasses)
  if (glasses > 0.2) {
    const alpha = Math.min(1, glasses);
    ctx.fillStyle = `rgba(15, 23, 42, ${alpha * 0.9})`;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    // Mắt kính trái & phải
    ctx.beginPath();
    ctx.roundRect(w/2 - 45, eyeY - 14, 34, 28, 6);
    ctx.roundRect(w/2 + 11, eyeY - 14, 34, 28, 6);
    ctx.fill();
    ctx.stroke();
    // Gọng kính nối
    ctx.beginPath(); ctx.moveTo(w/2 - 11, eyeY); ctx.lineTo(w/2 + 11, eyeY); ctx.stroke();
  }

  // Cập nhật công thức vector đại số hiển thị
  const formulaEl = document.getElementById('latent-vector-formula');
  if (formulaEl) {
    const smileSign = smile >= 0 ? `+ ${(smile).toFixed(1)}` : `- ${Math.abs(smile).toFixed(1)}`;
    formulaEl.innerHTML = `$$\\mathbf{w}_{out} = \\mathbf{w}_{base} ${smileSign}\\cdot\\vec{v}_{cười} + ${(glasses).toFixed(1)}\\cdot\\vec{v}_{kính} + \\left(\\frac{${age}-25}{50}\\right)\\cdot\\vec{v}_{tuổi}$$`;
    if (window.renderMathInElement) window.renderMathInElement(formulaEl);
  }
}

// ==========================================
// 4. MINI-GAME: THÁM TỬ AI (TURING TEST)
// ==========================================
const TuringQuestions = [
  {
    title: "Vòng 1: Đôi mắt và Bông tai",
    desc: "Hãy chú ý độ đối xứng của hai bên bông tai và con ngươi trong mắt!",
    imgReal: "assets/turing_real_1.svg",
    imgFake: "assets/turing_fake_1.svg",
    fakeSide: "B", // A or B
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
        <p class="text-slate-400 max-w-md mx-auto text-sm">Bạn đã hiểu được cách mà mạng Discriminator liên tục soi các chi tiết bất thường (bông tai, nếp tóc, ngón tay) để bắt bài Generator!</p>
        <button onclick="userScore=0; loadTuringQuestion(0);" class="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold rounded-xl transition shadow-lg shadow-cyan-500/20">Chơi Lại</button>
      </div>
    `;
  }
}
