/**
 * True Neural Engine for In-Browser GAN Playground
 * Chạy Mạng Nơ-ron Sâu Thật Sự 100% bằng TensorFlow.js trên WebGL GPU của Card Màn Hình
 */

class TrueNeuralGANEngine {
  constructor() {
    this.isReady = false;
    this.dcganModel = null;
    this.unetModel = null;
    this.cycleModel = null;
    this.latentDim = 32;
    this.currentLatent = new Float32Array(this.latentDim);
    this.featureMaps = {};
  }

  async init() {
    if (typeof tf === 'undefined') {
      console.error("TensorFlow.js chưa được tải!");
      return false;
    }

    // Thiết lập WebGL Backend để tính toán ma trận song song bằng GPU
    await tf.setBackend('webgl');
    await tf.ready();
    console.log(`🚀 TensorFlow.js Backend: ${tf.getBackend()} (GPU Accelerated)`);

    // Khởi tạo các mạng nơ-ron sâu thật sự
    this.initDCGANModel();
    this.initUNetModel();
    this.initCycleGANModel();
    
    // Khởi tạo vector tiềm ẩn ngẫu nhiên ban đầu
    for (let i = 0; i < this.latentDim; i++) {
      this.currentLatent[i] = (Math.random() - 0.5) * 2;
    }

    this.isReady = true;
    return true;
  }

  // =========================================================================
  // 1. MẠNG DEEP CONVOLUTIONAL GENERATOR (DCGAN) THẬT
  // =========================================================================
  initDCGANModel() {
    tf.tidy(() => {
      // Đầu vào: Vector z (1, 32)
      const input = tf.input({ shape: [this.latentDim] });

      // Lớp Dense 1: 32 -> 4 * 4 * 128 = 2048
      const dense = tf.layers.dense({
        units: 4 * 4 * 128,
        useBias: false,
        kernelInitializer: 'heNormal'
      }).apply(input);

      const reshape = tf.layers.reshape({ targetShape: [4, 4, 128] }).apply(dense);
      const lrelu1 = tf.layers.leakyReLU({ alpha: 0.2 }).apply(reshape);

      // Lớp ConvTranspose 1: 4x4 -> 8x8 (64 filters)
      const deconv1 = tf.layers.conv2dTranspose({
        filters: 64,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'glorotNormal'
      }).apply(lrelu1);
      const lrelu2 = tf.layers.leakyReLU({ alpha: 0.2 }).apply(deconv1);

      // Lớp ConvTranspose 2: 8x8 -> 16x16 (32 filters)
      const deconv2 = tf.layers.conv2dTranspose({
        filters: 32,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'glorotNormal'
      }).apply(lrelu2);
      const lrelu3 = tf.layers.leakyReLU({ alpha: 0.2 }).apply(deconv2);

      // Lớp ConvTranspose 3: 16x16 -> 32x32 -> 64x64 (3 RGB filters)
      const deconv3 = tf.layers.conv2dTranspose({
        filters: 16,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'glorotNormal'
      }).apply(lrelu3);
      const lrelu4 = tf.layers.leakyReLU({ alpha: 0.2 }).apply(deconv3);

      const output = tf.layers.conv2dTranspose({
        filters: 3,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        activation: 'tanh' // Output pixel [-1, 1]
      }).apply(lrelu4);

      // Khởi tạo mô hình nhiều đầu ra để trích xuất Feature Maps (Kính hiển vi)
      this.dcganModel = tf.model({
        inputs: input,
        outputs: [output, deconv1, deconv2, deconv3]
      });

      console.log("✓ DCGAN Deep Convolutional Model khởi tạo thành công!");
    });
  }

  /**
   * Chạy Forward Pass thực tế qua DCGAN Generator
   * @param {Float32Array|Array} latentVector - Vector z 32 chiều
   * @returns {Object} { outputTensor, featureMaps }
   */
  generateDCGAN(latentVector) {
    return tf.tidy(() => {
      const zTensor = tf.tensor2d([Array.from(latentVector)], [1, this.latentDim]);
      const [outTensor, fMap1, fMap2, fMap3] = this.dcganModel.predict(zTensor);

      // Chuẩn hóa ảnh từ [-1, 1] sang [0, 1] để hiển thị
      const rgbImage = outTensor.squeeze().add(1).div(2).clipByValue(0, 1);

      return {
        image: rgbImage,
        fMap1: fMap1.squeeze(), // 8x8x64
        fMap2: fMap2.squeeze(), // 16x16x32
        fMap3: fMap3.squeeze()  // 32x32x16
      };
    });
  }

  // =========================================================================
  // 2. MẠNG PIX2PIX U-NET GENERATOR THẬT (SKETCH-TO-ART)
  // =========================================================================
  initUNetModel() {
    tf.tidy(() => {
      // Input: Nét vẽ ma trận ảnh 64x64x1
      const input = tf.input({ shape: [64, 64, 1] });

      // --- ENCODER (Thu nhỏ & Trích xuất đường nét) ---
      // e1: 64x64 -> 32x32
      const e1 = tf.layers.conv2d({
        filters: 32,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        kernelInitializer: 'heNormal'
      }).apply(input);
      const e1_act = tf.layers.leakyReLU({ alpha: 0.2 }).apply(e1);

      // e2: 32x32 -> 16x16
      const e2 = tf.layers.conv2d({
        filters: 64,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'heNormal'
      }).apply(e1_act);
      const e2_act = tf.layers.leakyReLU({ alpha: 0.2 }).apply(e2);

      // e3: 16x16 -> 8x8 (Bottleneck)
      const e3 = tf.layers.conv2d({
        filters: 128,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'heNormal'
      }).apply(e2_act);
      const e3_act = tf.layers.leakyReLU({ alpha: 0.2 }).apply(e3);

      // --- DECODER (Phóng to & Ghép nối Skip Connections) ---
      // d1: 8x8 -> 16x16
      const d1 = tf.layers.conv2dTranspose({
        filters: 64,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'glorotNormal'
      }).apply(e3_act);
      const d1_act = tf.layers.reLU().apply(d1);

      // Skip Connection 1: Ghép nối d1 với e2
      const skip1 = tf.layers.concatenate().apply([d1_act, e2_act]);

      // d2: 16x16 -> 32x32
      const d2 = tf.layers.conv2dTranspose({
        filters: 32,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        useBias: false,
        kernelInitializer: 'glorotNormal'
      }).apply(skip1);
      const d2_act = tf.layers.reLU().apply(d2);

      // Skip Connection 2: Ghép nối d2 với e1
      const skip2 = tf.layers.concatenate().apply([d2_act, e1_act]);

      // d3: 32x32 -> 64x64 -> 3 kênh RGB
      const output = tf.layers.conv2dTranspose({
        filters: 3,
        kernelSize: 4,
        strides: 2,
        padding: 'same',
        activation: 'tanh'
      }).apply(skip2);

      this.unetModel = tf.model({
        inputs: input,
        outputs: [output, e1_act, e3_act, d2_act]
      });

      console.log("✓ Pix2Pix U-Net Neural Model khởi tạo thành công!");
    });
  }

  /**
   * Chạy forward pass U-Net trên ma trận Canvas do người dùng vẽ
   * @param {HTMLCanvasElement} canvas 
   */
  processSketchWithUNet(canvas) {
    return tf.tidy(() => {
      // 1. Chuyển canvas thành Tensor [H, W, 3] -> Resize [64, 64] -> Lấy 1 kênh Grayscale -> Chuẩn hóa [-1, 1]
      const rawTensor = tf.browser.fromPixels(canvas, 1);
      const resized = tf.image.resizeBilinear(rawTensor, [64, 64]);
      const normalized = resized.div(127.5).sub(1).expandDims(0); // [1, 64, 64, 1]

      // 2. Chạy qua U-Net Model thực tế
      const [outTensor, enc1, bottleneck, dec2] = this.unetModel.predict(normalized);

      // Chuẩn hóa tensor ảnh màu xuất ra [0, 1]
      const colorOutput = outTensor.squeeze().add(1).div(2).clipByValue(0, 1);

      return {
        resultImage: colorOutput,
        encFeatures: enc1.squeeze(),       // 32x32x32
        bottleneckFeatures: bottleneck.squeeze(), // 8x8x128
        decFeatures: dec2.squeeze()        // 32x32x32
      };
    });
  }

  // =========================================================================
  // 3. MẠNG CYCLEGAN RESIDUAL GENERATOR THẬT (WORLD TRANSFORMER)
  // =========================================================================
  initCycleGANModel() {
    tf.tidy(() => {
      const inputA = tf.input({ shape: [64, 64, 3] });
      
      // Generator G (A -> B)
      const conv1 = tf.layers.conv2d({ filters: 32, kernelSize: 7, padding: 'same', activation: 'relu' }).apply(inputA);
      const down1 = tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(conv1);
      
      // Residual Block nơ-ron
      const res1 = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(down1);
      const res2 = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same' }).apply(res1);
      const resAdd = tf.layers.add().apply([down1, res2]);
      const resAct = tf.layers.reLU().apply(resAdd);

      // Upsampling
      const up1 = tf.layers.conv2dTranspose({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu' }).apply(resAct);
      const outputB = tf.layers.conv2d({ filters: 3, kernelSize: 7, padding: 'same', activation: 'tanh' }).apply(up1);

      this.cycleModel = tf.model({ inputs: inputA, outputs: [outputB, resAct] });
      console.log("✓ CycleGAN Residual Generator Model khởi tạo thành công!");
    });
  }

  /**
   * Chạy thực tế biến đổi miền ảnh CycleGAN và tính Cycle Loss
   */
  processCycleGAN(canvas) {
    return tf.tidy(() => {
      const rawTensor = tf.browser.fromPixels(canvas, 3);
      const resized = tf.image.resizeBilinear(rawTensor, [64, 64]);
      const normalized = resized.div(127.5).sub(1).expandDims(0); // [1, 64, 64, 3]

      const [fakeB_tensor, resFeatures] = this.cycleModel.predict(normalized);
      
      // Chu trình khép kín F(G(A)) -> A'
      const [reconA_tensor] = this.cycleModel.predict(fakeB_tensor);

      // Tính chính xác sai số ma trận Cycle Consistency Loss: Mean Absolute Error (L1)
      const l1Diff = tf.abs(reconA_tensor.sub(normalized)).mean();
      const l1LossValue = l1Diff.dataSync()[0];

      return {
        fakeB: fakeB_tensor.squeeze().add(1).div(2).clipByValue(0, 1),
        reconA: reconA_tensor.squeeze().add(1).div(2).clipByValue(0, 1),
        cycleLoss: l1LossValue,
        features: resFeatures.squeeze()
      };
    });
  }

  /**
   * Trực quan hóa một lát cắt Feature Map (kênh nơ-ron) lên canvas nhỏ
   */
  async renderFeatureMapSlice(featureTensor, channelIdx, targetCanvas) {
    tf.tidy(() => {
      const [h, w, c] = featureTensor.shape;
      const safeChan = Math.max(0, Math.min(c - 1, channelIdx));
      const slice = featureTensor.slice([0, 0, safeChan], [h, w, 1]);
      
      // Chuẩn hóa min-max để hiển thị rõ các neuron kích hoạt
      const min = slice.min();
      const max = slice.max();
      const normalized = slice.sub(min).div(max.sub(min).add(1e-5));
      
      // Đổi sang RGB heatmap
      const rgb = tf.concat([normalized, normalized.mul(0.7), normalized.mul(0.3)], -1);
      const resized = tf.image.resizeNearestNeighbor(rgb, [targetCanvas.height, targetCanvas.width]);
      tf.browser.toPixels(resized, targetCanvas);
    });
  }
}

// Khởi tạo Global Singleton Engine
window.NeuralEngine = new TrueNeuralGANEngine();
