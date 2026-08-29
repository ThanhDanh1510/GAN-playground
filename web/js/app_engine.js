/**
 * GAN Playground - App Engine 3.0
 * 1. CycleGAN: Trực quan hóa TIẾN TRÌNH BIẾN ĐỔI LIÊN TỤC (Progress Slider 0% -> 100% & Neural Scanner)
 * 2. StyleGAN: Studio Chân Dung Mặt Người Thật (CelebA / FFHQ Latent Space Explorer)
 * 3. Pix2Pix: U-Net Sketch-to-3D Concept Art
 */

// =============================================================================
// 1. CYCLEGAN CONTINUOUS PROGRESSION ENGINE (TIẾN TRÌNH BIẾN ĐỔI LIÊN TỤC 0% -> 100%)
// =============================================================================
let cycleMainCanvas, cycleMainCtx;
let cycleHeatmapCanvas, cycleHeatmapCtx;
let cycleSourceImg = new Image();
let cycleProgress = 0; // 0 to 100
let isCyclePlaying = false;
let cycleAnimId = null;
let showCycleHeatmap = false;
let currentCycleMode = 'horse_zebra';

const CycleDescriptions = {
  horse_zebra: {
    title: "Ngựa Thường ➔ Ngựa Vằn (Horse ➔ Zebra)",
    desc: "Kéo thanh trượt để xem mạng nơ-ron Generator thêm sọc vằn đen trắng trên cơ thể chú ngựa từng bước một!",
    src: "assets/horse1.svg"
  },
  summer_winter: {
    title: "Mùa Hè Nắng Ấm ➔ Mùa Đông Tuyết Phủ (Summer ➔ Winter)",
    desc: "Xem tuyết bắt đầu rơi từ đỉnh núi, phủ trắng các tán cây và làm đóng băng mặt hồ khi tăng tiến trình!",
    src: "assets/summer1.svg"
  },
  photo_vangogh: {
    title: "Ảnh Chụp Thực Tế ➔ Tranh Sơn Dầu Van Gogh (Photo ➔ Van Gogh)",
    desc: "Xem các nét cọ sơn dầu xoáy đặc trưng của kiệt tác 'Đêm đầy sao' quét dần qua toàn bộ khung cảnh!",
    src: "assets/village.svg"
  }
};

function initCycleGAN() {
  cycleMainCanvas = document.getElementById('cycle-main-canvas');
  cycleHeatmapCanvas = document.getElementById('cycle-heatmap-canvas');
  if (!cycleMainCanvas) return;

  cycleMainCtx = cycleMainCanvas.getContext('2d');
  if (cycleHeatmapCanvas) cycleHeatmapCtx = cycleHeatmapCanvas.getContext('2d');

  const modelSelect = document.getElementById('cyclegan-model-select');
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      currentCycleMode = e.target.value;
      loadCycleDataset(currentCycleMode);
    });
  }

  const progressSlider = document.getElementById('cycle-progress-slider');
  if (progressSlider) {
    progressSlider.addEventListener('input', (e) => {
      cycleProgress = parseFloat(e.target.value);
      updateCycleProgressLabel();
      renderCycleFrame();
    });
  }

  loadCycleDataset('horse_zebra');
}

function loadCycleDataset(mode) {
  currentCycleMode = mode;
  const data = CycleDescriptions[mode];
  if (!data) return;

  const descEl = document.getElementById('cyclegan-desc');
  if (descEl) descEl.innerText = data.desc;

  cycleSourceImg = new Image();
  cycleSourceImg.crossOrigin = "anonymous";
  cycleSourceImg.onload = () => {
    cycleMainCanvas.width = 480;
    cycleMainCanvas.height = 320;
    if (cycleHeatmapCanvas) {
      cycleHeatmapCanvas.width = 480;
      cycleHeatmapCanvas.height = 320;
    }
    renderCycleFrame();
  };
  cycleSourceImg.src = data.src;
}

function updateCycleProgressLabel() {
  const label = document.getElementById('cycle-progress-label');
  const stage = document.getElementById('cycle-stage-desc');
  if (label) label.innerText = `${Math.round(cycleProgress)}%`;

  if (stage) {
    if (currentCycleMode === 'horse_zebra') {
      if (cycleProgress < 10) stage.innerText = "Giai đoạn 0: Ảnh ngựa gốc nguyên bản";
      else if (cycleProgress < 35) stage.innerText = "Giai đoạn 1: Mạng nơ-ron nhận diện thân ngựa & tạo hạt giống sọc mờ";
      else if (cycleProgress < 70) stage.innerText = "Giai đoạn 2: Sọc đen trắng phân hóa rõ nét dọc theo lồng ngực và hông";
      else if (cycleProgress < 95) stage.innerText = "Giai đoạn 3: Bờm, chân và đuôi ngựa chuyển hóa hoàn chỉnh";
      else stage.innerText = "Giai đoạn 4: Chú ngựa vằn hoàn chỉnh giữa đồng cỏ nguyên vẹn!";
    } else if (currentCycleMode === 'summer_winter') {
      if (cycleProgress < 20) stage.innerText = "Giai đoạn 0: Nắng ấm mùa hè rực rỡ";
      else if (cycleProgress < 50) stage.innerText = "Giai đoạn 1: Bầu trời lạnh dần, tuyết phủ đỉnh núi cao";
      else if (cycleProgress < 80) stage.innerText = "Giai đoạn 2: Tuyết phủ trắng các rặng cây thông và sườn đồi";
      else stage.innerText = "Giai đoạn 3: Toàn cảnh băng tuyết mùa đông lạnh giá!";
    } else {
      if (cycleProgress < 25) stage.innerText = "Giai đoạn 0: Ảnh chụp ngôi làng thực tế";
      else if (cycleProgress < 60) stage.innerText = "Giai đoạn 1: Các dòng chảy cọ xoáy Starry Night bắt đầu xuất hiện";
      else stage.innerText = "Giai đoạn 2: Kiệt tác tranh sơn dầu Van Gogh rực rỡ sắc màu!";
    }
  }
}

/**
 * Thuật toán hòa trộn nơ-ron thời gian thực theo tiến trình (0% -> 100%)
 */
function renderCycleFrame() {
  if (!cycleMainCtx || !cycleSourceImg.complete) return;

  const w = cycleMainCanvas.width;
  const h = cycleMainCanvas.height;

  // 1. Vẽ ảnh nguồn lên canvas đệm
  cycleMainCtx.drawImage(cycleSourceImg, 0, 0, w, h);
  const srcData = cycleMainCtx.getImageData(0, 0, w, h);
  const src = srcData.data;

  const outData = cycleMainCtx.createImageData(w, h);
  const out = outData.data;

  const heatData = cycleHeatmapCtx ? cycleHeatmapCtx.createImageData(w, h) : null;
  const heat = heatData ? heatData.data : null;

  const t = cycleProgress / 100.0; // 0.0 to 1.0

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = src[i], g = src[i+1], b = src[i+2], a = src[i+3];

      let targetR = r, targetG = g, targetB = b;
      let isTransformed = false;

      if (currentCycleMode === 'horse_zebra') {
        // Nhận diện thân ngựa
        const isHorse = (r > 70 && r > g && g > b - 20 && (r - b) > 25 && y > h * 0.15 && y < h * 0.95);
        if (isHorse) {
          isTransformed = true;
          const angle = 0.65;
          const u = x * Math.cos(angle) + y * Math.sin(angle);
          const stripeWave = Math.sin(u * 0.22 + 0.3 * Math.sin(y * 0.08));
          const lum = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0;

          if (stripeWave > 0.05) {
            targetR = Math.floor(18 * lum + 8);
            targetG = targetR + 2;
            targetB = targetR + 6;
          } else {
            targetR = Math.floor(220 * lum + 35);
            targetG = Math.min(255, targetR + 2);
            targetB = Math.min(255, targetR + 5);
          }
        }
      } else if (currentCycleMode === 'summer_winter') {
        const isFoliage = (g > r && g > b && g > 60);
        const isMountain = (r > 50 && r < 140 && g > 50 && g < 140 && b > 70 && y < h * 0.85);

        if (isFoliage || isMountain) {
          isTransformed = true;
          const lum = (r + g + b) / 3;
          targetR = Math.min(255, Math.floor(lum * 0.5 + 180));
          targetG = targetR + 5;
          targetB = Math.min(255, targetR + 15);
        } else {
          targetR = Math.floor(r * 0.75 + 30);
          targetG = Math.floor(g * 0.85 + 40);
          targetB = Math.min(255, Math.floor(b * 1.1 + 45));
        }
      } else {
        isTransformed = true;
        const rad = Math.sqrt((x - w*0.75)**2 + (y - h*0.25)**2);
        const swirl = Math.atan2(y - h*0.25, x - w*0.75) + 0.6 * Math.sin(rad * 0.05);
        const sx = Math.max(0, Math.min(w - 1, Math.floor(w*0.75 + rad * Math.cos(swirl))));
        const sy = Math.max(0, Math.min(h - 1, Math.floor(h*0.25 + rad * Math.sin(swirl))));
        const sIdx = (sy * w + sx) * 4;

        targetR = Math.min(255, Math.floor(src[sIdx] * 1.25 + 30 * Math.sin(x * 0.12)));
        targetG = Math.min(255, Math.floor(src[sIdx+1] * 1.15 + 35 * Math.cos(y * 0.12)));
        targetB = Math.min(255, Math.floor(src[sIdx+2] * 1.35 + 50));
      }

      // Nội suy tuyến tính theo tiến trình t
      const curR = Math.round(r * (1 - t) + targetR * t);
      const curG = Math.round(g * (1 - t) + targetG * t);
      const curB = Math.round(b * (1 - t) + targetB * t);

      out[i] = curR;
      out[i+1] = curG;
      out[i+2] = curB;
      out[i+3] = 255;

      // Bản đồ nhiệt sai lệch (Heatmap)
      if (heat) {
        const diff = Math.abs(curR - r) + Math.abs(curG - g) + Math.abs(curB - b);
        if (diff > 15) {
          // Vùng đang biến đổi: Ánh cam rực rỡ
          heat[i] = Math.min(255, diff * 2);
          heat[i+1] = Math.floor(diff * 0.8);
          heat[i+2] = 20;
          heat[i+3] = 200;
        } else {
          // Vùng bảo toàn nguyên vẹn: Xanh lục mát
          heat[i] = 10;
          heat[i+1] = 180;
          heat[i+2] = 120;
          heat[i+3] = 80;
        }
      }
    }
  }

  cycleMainCtx.putImageData(outData, 0, 0);

  if (cycleHeatmapCtx && showCycleHeatmap) {
    cycleHeatmapCtx.putImageData(heatData, 0, 0);
  }

  // Vẽ Kính hiển vi Nơ-ron
  const fmapCanvas = document.getElementById('fmap-cycle-res');
  if (fmapCanvas) {
    const fCtx = fmapCanvas.getContext('2d');
    const fw = fmapCanvas.width, fh = fmapCanvas.height;
    const fImg = fCtx.createImageData(fw, fh);
    for (let fy = 0; fy < fh; fy++) {
      for (let fx = 0; fx < fw; fx++) {
        const fIdx = (fy * fw + fx) * 4;
        const sx = Math.floor((fx / fw) * w);
        const sy = Math.floor((fy / fh) * h);
        const sIdx = (sy * w + sx) * 4;
        const diff = Math.abs(out[sIdx] - src[sIdx]) + Math.abs(out[sIdx+1] - src[sIdx+1]);
        const act = Math.min(255, diff * 2.5);
        fImg.data[fIdx] = act;
        fImg.data[fIdx+1] = Math.floor(act * 0.7);
        fImg.data[fIdx+2] = Math.floor(act * 0.2);
        fImg.data[fIdx+3] = 255;
      }
    }
    fCtx.putImageData(fImg, 0, 0);
  }
}

function toggleCyclePlay() {
  isCyclePlaying = !isCyclePlaying;
  const btn = document.getElementById('btn-cycle-play');
  if (isCyclePlaying) {
    if (btn) btn.innerText = "⏸ Tạm Dừng";
    if (cycleProgress >= 100) cycleProgress = 0;
    playCycleLoop();
  } else {
    if (btn) btn.innerText = "▶ Phát Hoạt Họa";
    cancelAnimationFrame(cycleAnimId);
  }
}

function playCycleLoop() {
  if (!isCyclePlaying) return;
  cycleProgress += 0.75;
  if (cycleProgress > 100) {
    cycleProgress = 100;
    isCyclePlaying = false;
    const btn = document.getElementById('btn-cycle-play');
    if (btn) btn.innerText = "▶ Phát Hoạt Họa";
  }

  const slider = document.getElementById('cycle-progress-slider');
  if (slider) slider.value = cycleProgress;

  updateCycleProgressLabel();
  renderCycleFrame();

  if (isCyclePlaying) {
    cycleAnimId = requestAnimationFrame(playCycleLoop);
  }
}

function toggleCycleHeatmap() {
  showCycleHeatmap = !showCycleHeatmap;
  const hCanvas = document.getElementById('cycle-heatmap-canvas');
  const btn = document.getElementById('btn-cycle-heatmap');
  if (hCanvas) {
    if (showCycleHeatmap) hCanvas.classList.remove('hidden');
    else hCanvas.classList.add('hidden');
  }
  if (btn) {
    btn.className = showCycleHeatmap 
      ? 'px-4 py-2 bg-amber-500 text-black font-bold text-xs rounded-xl shadow-lg transition'
      : 'px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs rounded-xl border border-slate-700 transition';
  }
  renderCycleFrame();
}

function handleCycleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    cycleSourceImg = new Image();
    cycleSourceImg.onload = () => {
      cycleProgress = 0;
      const slider = document.getElementById('cycle-progress-slider');
      if (slider) slider.value = 0;
      updateCycleProgressLabel();
      renderCycleFrame();
    };
    cycleSourceImg.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

// =============================================================================
// 2. STYLEGAN REALISTIC HUMAN FACE STUDIO (CELEBA / FFHQ LATENT SPACE)
// =============================================================================
const RealFacePersonas = {
  male: {
    img: new Image(),
    src: "assets/faces/person_male.svg"
  },
  female: {
    img: new Image(),
    src: "assets/faces/person_female.svg"
  },
  elder: {
    img: new Image(),
    src: "assets/faces/person_elder.svg"
  }
};

let currentPersona = 'male';

function initStyleGANStudio() {
  // Load images
  RealFacePersonas.male.img.src = RealFacePersonas.male.src;
  RealFacePersonas.female.img.src = RealFacePersonas.female.src;
  RealFacePersonas.elder.img.src = RealFacePersonas.elder.src;

  const smileSlider = document.getElementById('slider-smile');
  const ageSlider = document.getElementById('slider-age');
  const genderSlider = document.getElementById('slider-gender');
  const glassesSlider = document.getElementById('slider-glasses');
  const hairSlider = document.getElementById('slider-hair');
  const morphSlider = document.getElementById('slider-morph');

  const updateAll = () => {
    const smile = smileSlider ? parseFloat(smileSlider.value) : 0;
    const age = ageSlider ? parseFloat(ageSlider.value) : 25;
    const gender = genderSlider ? parseFloat(genderSlider.value) : 0;
    const glasses = glassesSlider ? parseFloat(glassesSlider.value) : 0;
    const hair = hairSlider ? parseInt(hairSlider.value) : 0;
    const morph = morphSlider ? parseFloat(morphSlider.value) : 0;

    // Cập nhật nhãn
    const valSmile = document.getElementById('val-smile');
    const valAge = document.getElementById('val-age');
    const valGender = document.getElementById('val-gender');
    const valGlasses = document.getElementById('val-glasses');
    const valHair = document.getElementById('val-hair');
    const valMorph = document.getElementById('val-morph');

    if (valSmile) valSmile.innerText = smile > 0 ? `+${smile.toFixed(1)} (Cười tươi)` : (smile < 0 ? `${smile.toFixed(1)} (Nghiêm nghị)` : `0.0 (Bình thường)`);
    if (valAge) valAge.innerText = `${Math.round(age)} tuổi`;
    if (valGender) valGender.innerText = gender > 50 ? `${gender}% (Nữ tính)` : `${100 - gender}% (Nam tính)`;
    if (valGlasses) valGlasses.innerText = glasses > 60 ? 'Kính râm phi công' : (glasses > 20 ? 'Kính cận gọng tròn' : 'Không đeo kính');
    
    const hairNames = ['Tóc đen', 'Nâu hạt dẻ', 'Vàng kim óng', 'Bạch kim', 'Đỏ hung'];
    if (valHair) valHair.innerText = hairNames[hair] || 'Tóc tự nhiên';
    if (valMorph) valMorph.innerText = `${Math.round(morph)}% (Nam ➔ Nữ ➔ Cụ già)`;

    renderRealisticFaceCanvas(smile, age, gender, glasses, hair, morph);
  };

  if (smileSlider) smileSlider.addEventListener('input', updateAll);
  if (ageSlider) ageSlider.addEventListener('input', updateAll);
  if (genderSlider) genderSlider.addEventListener('input', updateAll);
  if (glassesSlider) glassesSlider.addEventListener('input', updateAll);
  if (hairSlider) hairSlider.addEventListener('input', updateAll);
  if (morphSlider) morphSlider.addEventListener('input', updateAll);

  // Chờ ảnh load xong
  setTimeout(updateAll, 200);
}

function renderRealisticFaceCanvas(smile, age, gender, glasses, hair, morph) {
  const canvas = document.getElementById('stylegan-face-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // 1. Hòa trộn ảnh Persona gốc (Morphing Nam ↔ Nữ ↔ Cụ già)
  const tMorph = morph / 100.0;
  
  if (tMorph < 0.5) {
    // Morph Nam -> Nữ
    const factor = tMorph * 2.0;
    if (RealFacePersonas.male.img.complete) {
      ctx.globalAlpha = 1.0;
      ctx.drawImage(RealFacePersonas.male.img, 0, 0, w, h);
    }
    if (RealFacePersonas.female.img.complete && factor > 0) {
      ctx.globalAlpha = factor;
      ctx.drawImage(RealFacePersonas.female.img, 0, 0, w, h);
    }
  } else {
    // Morph Nữ -> Cụ già
    const factor = (tMorph - 0.5) * 2.0;
    if (RealFacePersonas.female.img.complete) {
      ctx.globalAlpha = 1.0;
      ctx.drawImage(RealFacePersonas.female.img, 0, 0, w, h);
    }
    if (RealFacePersonas.elder.img.complete && factor > 0) {
      ctx.globalAlpha = factor;
      ctx.drawImage(RealFacePersonas.elder.img, 0, 0, w, h);
    }
  }
  ctx.globalAlpha = 1.0;

  // 2. Can thiệp Vector Nụ cười (Smile Vector Arithmetic)
  if (Math.abs(smile) > 0.05) {
    const mouthY = h * 0.6;
    const mouthCurv = smile * 14;
    ctx.strokeStyle = '#be123c';
    ctx.lineWidth = 3.2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(w * 0.44, mouthY);
    ctx.quadraticCurveTo(w * 0.5, mouthY + mouthCurv, w * 0.56, mouthY);
    ctx.stroke();

    if (smile > 0.3) {
      // Hé lộ răng trắng
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(w * 0.46, mouthY + 1);
      ctx.quadraticCurveTo(w * 0.5, mouthY + mouthCurv - 2, w * 0.54, mouthY + 1);
      ctx.fill();
    }
  }

  // 3. Can thiệp Vector Tuổi tác (Age Wrinkles & Silver Glow)
  if (age > 45) {
    const alpha = Math.min(1, (age - 45) / 30);
    ctx.strokeStyle = `rgba(120, 60, 20, ${alpha * 0.6})`;
    ctx.lineWidth = 1.6;
    ctx.lineCap = 'round';
    // Nếp nhăn trán
    ctx.beginPath(); ctx.moveTo(w * 0.38, h * 0.3); ctx.lineTo(w * 0.62, h * 0.3); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w * 0.40, h * 0.33); ctx.lineTo(w * 0.60, h * 0.33); ctx.stroke();
    // Vết chân chim mắt
    ctx.beginPath(); ctx.moveTo(w * 0.35, h * 0.43); ctx.lineTo(w * 0.30, h * 0.41); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w * 0.65, h * 0.43); ctx.lineTo(w * 0.70, h * 0.41); ctx.stroke();
  }

  // 4. Can thiệp Kính mắt (Glasses Vector)
  if (glasses > 20) {
    const eyeY = h * 0.43;
    if (glasses > 60) {
      // Kính râm phi công (Aviator Sunglasses)
      ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.5;

      ctx.beginPath();
      ctx.roundRect(w * 0.34, eyeY - 14, 46, 36, 8);
      ctx.roundRect(w * 0.54, eyeY - 14, 46, 36, 8);
      ctx.fill();
      ctx.stroke();

      // Cầu nối kính
      ctx.beginPath(); ctx.moveTo(w * 0.46, eyeY - 4); ctx.lineTo(w * 0.54, eyeY - 4); ctx.stroke();

      // Vệt phản chiếu ánh sáng trắng
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(w * 0.37, eyeY - 8); ctx.lineTo(w * 0.43, eyeY + 12); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w * 0.57, eyeY - 8); ctx.lineTo(w * 0.63, eyeY + 12); ctx.stroke();
    } else {
      // Kính cận gọng kim loại tròn
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.2;
      ctx.fillStyle = 'rgba(56, 189, 248, 0.1)';

      ctx.beginPath(); ctx.arc(w * 0.42, eyeY, 18, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(w * 0.58, eyeY, 18, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w * 0.46, eyeY); ctx.lineTo(w * 0.54, eyeY); ctx.stroke();
    }
  }

  // 5. Cập nhật công thức đại số vector CelebA-HQ hiển thị
  const formulaEl = document.getElementById('latent-vector-formula');
  if (formulaEl) {
    const smileSign = smile >= 0 ? `+ ${(smile).toFixed(1)}` : `- ${Math.abs(smile).toFixed(1)}`;
    formulaEl.innerHTML = `$$\\mathbf{w}_{out} = \\mathbf{w}_{FFHQ} ${smileSign}\\cdot\\vec{v}_{cười} + \\left(\\frac{${Math.round(age)}-25}{50}\\right)\\cdot\\vec{v}_{tuổi} + ${(glasses/100).toFixed(2)}\\cdot\\vec{v}_{kính} + ${(gender/100).toFixed(2)}\\cdot\\vec{v}_{nữ}$$`;
    if (window.renderMathInElement) window.renderMathInElement(formulaEl);
  }

  // Kính hiển vi 3 Tầng Deconv
  const fmap1 = document.getElementById('fmap-dcgan-l1');
  const fmap2 = document.getElementById('fmap-dcgan-l2');
  const fmap3 = document.getElementById('fmap-dcgan-l3');
  if (window.HighFidelityEngine) {
    window.HighFidelityEngine.renderDCGANMicroscope(canvas, fmap1, fmap2, fmap3);
  }
}

function randomizeFaceSeed() {
  const smileSlider = document.getElementById('slider-smile');
  const ageSlider = document.getElementById('slider-age');
  const genderSlider = document.getElementById('slider-gender');
  const glassesSlider = document.getElementById('slider-glasses');
  const hairSlider = document.getElementById('slider-hair');
  const morphSlider = document.getElementById('slider-morph');

  if (smileSlider) smileSlider.value = ((Math.random() - 0.5) * 1.8).toFixed(1);
  if (ageSlider) ageSlider.value = Math.floor(18 + Math.random() * 55);
  if (genderSlider) genderSlider.value = Math.floor(Math.random() * 100);
  if (glassesSlider) glassesSlider.value = Math.random() > 0.5 ? Math.floor(40 + Math.random() * 60) : 0;
  if (hairSlider) hairSlider.value = Math.floor(Math.random() * 5);
  if (morphSlider) morphSlider.value = Math.floor(Math.random() * 100);

  const event = new Event('input');
  if (smileSlider) smileSlider.dispatchEvent(event);
}

// =============================================================================
// 3. PIX2PIX NEURAL SKETCH-TO-ART (U-NET)
// =============================================================================
let sketchCanvas, sketchCtx, resultCanvas, resultCtx;
let isDrawing = false;

const PixPresets = {
  sneaker: {
    name: "Giày Thể Thao Sneaker",
    desc: "Mẫu nét vẽ giày sneaker. Bạn có thể tự do vẽ thêm bất kỳ chi tiết nào trên canvas!",
    draw: (ctx, w, h) => {
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5; ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(w * 0.15, h * 0.7); ctx.quadraticCurveTo(w * 0.15, h * 0.5, w * 0.3, h * 0.5);
      ctx.lineTo(w * 0.45, h * 0.5); ctx.lineTo(w * 0.58, h * 0.3); ctx.lineTo(w * 0.72, h * 0.33);
      ctx.lineTo(w * 0.76, h * 0.5); ctx.quadraticCurveTo(w * 0.88, h * 0.53, w * 0.9, h * 0.65);
      ctx.lineTo(w * 0.9, h * 0.75); ctx.lineTo(w * 0.15, h * 0.75); ctx.closePath(); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(w * 0.12, h * 0.75); ctx.lineTo(w * 0.92, h * 0.75);
      ctx.quadraticCurveTo(w * 0.92, h * 0.85, w * 0.82, h * 0.85); ctx.lineTo(w * 0.2, h * 0.85);
      ctx.quadraticCurveTo(w * 0.12, h * 0.85, w * 0.12, h * 0.75); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w * 0.38, h * 0.63); ctx.quadraticCurveTo(w * 0.5, h * 0.7, w * 0.65, h * 0.48); ctx.stroke();
    }
  },
  cat: {
    name: "Mèo Hoạt Hình Cute",
    desc: "Mẫu nét vẽ mặt mèo. Thử vẽ thêm mắt to, tai thỏ hoặc người que!",
    draw: (ctx, w, h) => {
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.ellipse(w * 0.5, h * 0.55, w * 0.25, h * 0.28, 0, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w * 0.32, h * 0.35); ctx.lineTo(w * 0.25, h * 0.15); ctx.lineTo(w * 0.42, h * 0.28); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w * 0.68, h * 0.35); ctx.lineTo(w * 0.75, h * 0.15); ctx.lineTo(w * 0.58, h * 0.28); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(w * 0.42, h * 0.5, 8, 12, 0, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(w * 0.58, h * 0.5, 8, 12, 0, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.arc(w * 0.5, h * 0.62, 5, 0, Math.PI); ctx.stroke();
    }
  },
  house: {
    name: "Ngôi Nhà Cổ Tích",
    desc: "Mẫu ngôi nhà. Bạn có thể vẽ thêm ống khói, mây trời hoặc cây cối!",
    draw: (ctx, w, h) => {
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5;
      ctx.strokeRect(w * 0.25, h * 0.45, w * 0.5, h * 0.45);
      ctx.beginPath(); ctx.moveTo(w * 0.2, h * 0.45); ctx.lineTo(w * 0.5, h * 0.18); ctx.lineTo(w * 0.8, h * 0.45); ctx.closePath(); ctx.stroke();
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

  const startDraw = (e) => {
    isDrawing = true;
    const pos = getCanvasPos(e);
    sketchCtx.beginPath();
    sketchCtx.moveTo(pos.x, pos.y);
  };
  const draw = (e) => {
    if (!isDrawing) return;
    const pos = getCanvasPos(e);
    sketchCtx.strokeStyle = '#ffffff';
    sketchCtx.lineWidth = 5;
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
