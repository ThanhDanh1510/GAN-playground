/**
 * Mini 2D GAN Engine in Pure JavaScript
 * Huấn luyện GAN học phân phối điểm 2D theo thời gian thực trên trình duyệt
 */

class MiniTensor {
  constructor(data, shape) {
    this.data = new Float32Array(data);
    this.shape = shape; // [rows, cols]
  }

  static zeros(rows, cols) {
    return new MiniTensor(new Float32Array(rows * cols), [rows, cols]);
  }

  static randn(rows, cols, mean = 0, std = 1) {
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i += 2) {
      // Box-Muller transform
      const u1 = Math.max(1e-7, Math.random());
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
      data[i] = z0 * std + mean;
      if (i + 1 < data.length) data[i + 1] = z1 * std + mean;
    }
    return new MiniTensor(data, [rows, cols]);
  }
}

class DenseLayer {
  constructor(inFeatures, outFeatures, activation = 'leaky_relu') {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.activation = activation;

    // Khởi tạo trọng số He / Xavier
    const scale = Math.sqrt(2.0 / inFeatures);
    this.weights = MiniTensor.randn(inFeatures, outFeatures, 0, scale);
    this.bias = MiniTensor.zeros(1, outFeatures);

    // Gradients
    this.gradWeights = MiniTensor.zeros(inFeatures, outFeatures);
    this.gradBias = MiniTensor.zeros(1, outFeatures);

    // Adam optimizer cache
    this.m_w = new Float32Array(inFeatures * outFeatures);
    this.v_w = new Float32Array(inFeatures * outFeatures);
    this.m_b = new Float32Array(outFeatures);
    this.v_b = new Float32Array(outFeatures);
  }

  forward(input) {
    this.input = input;
    const batchSize = input.shape[0];
    const out = MiniTensor.zeros(batchSize, this.outFeatures);

    for (let b = 0; b < batchSize; b++) {
      for (let j = 0; j < this.outFeatures; j++) {
        let sum = this.bias.data[j];
        for (let i = 0; i < this.inFeatures; i++) {
          sum += input.data[b * this.inFeatures + i] * this.weights.data[i * this.outFeatures + j];
        }
        out.data[b * this.outFeatures + j] = sum;
      }
    }

    this.linearOutput = out;
    this.output = MiniTensor.zeros(batchSize, this.outFeatures);

    // Activation
    for (let k = 0; k < out.data.length; k++) {
      const val = out.data[k];
      if (this.activation === 'leaky_relu') {
        this.output.data[k] = val > 0 ? val : 0.2 * val;
      } else if (this.activation === 'tanh') {
        this.output.data[k] = Math.tanh(val);
      } else if (this.activation === 'sigmoid') {
        this.output.data[k] = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, val))));
      } else {
        this.output.data[k] = val;
      }
    }
    return this.output;
  }

  backward(gradOutput) {
    const batchSize = this.input.shape[0];
    const gradLinear = MiniTensor.zeros(batchSize, this.outFeatures);

    // Đạo hàm hàm kích hoạt
    for (let k = 0; k < gradLinear.data.length; k++) {
      const g = gradOutput.data[k];
      const z = this.linearOutput.data[k];
      const a = this.output.data[k];
      if (this.activation === 'leaky_relu') {
        gradLinear.data[k] = g * (z > 0 ? 1 : 0.2);
      } else if (this.activation === 'tanh') {
        gradLinear.data[k] = g * (1 - a * a);
      } else if (this.activation === 'sigmoid') {
        gradLinear.data[k] = g * (a * (1 - a));
      } else {
        gradLinear.data[k] = g;
      }
    }

    // Gradient trọng số & bias
    this.gradWeights.data.fill(0);
    this.gradBias.data.fill(0);
    const gradInput = MiniTensor.zeros(batchSize, this.inFeatures);

    for (let b = 0; b < batchSize; b++) {
      for (let j = 0; j < this.outFeatures; j++) {
        const dL_dz = gradLinear.data[b * this.outFeatures + j];
        this.gradBias.data[j] += dL_dz;
        for (let i = 0; i < this.inFeatures; i++) {
          const x = this.input.data[b * this.inFeatures + i];
          this.gradWeights.data[i * this.outFeatures + j] += x * dL_dz;
          gradInput.data[b * this.inFeatures + i] += this.weights.data[i * this.outFeatures + j] * dL_dz;
        }
      }
    }

    // Chuẩn hóa theo batch size
    for (let k = 0; k < this.gradWeights.data.length; k++) this.gradWeights.data[k] /= batchSize;
    for (let k = 0; k < this.gradBias.data.length; k++) this.gradBias.data[k] /= batchSize;

    return gradInput;
  }

  updateAdam(lr = 0.005, beta1 = 0.5, beta2 = 0.999, eps = 1e-8, t = 1) {
    // Cập nhật Weights
    for (let i = 0; i < this.weights.data.length; i++) {
      const g = this.gradWeights.data[i];
      this.m_w[i] = beta1 * this.m_w[i] + (1 - beta1) * g;
      this.v_w[i] = beta2 * this.v_w[i] + (1 - beta2) * (g * g);
      const m_hat = this.m_w[i] / (1 - Math.pow(beta1, t));
      const v_hat = this.v_w[i] / (1 - Math.pow(beta2, t));
      this.weights.data[i] -= (lr * m_hat) / (Math.sqrt(v_hat) + eps);
    }
    // Cập nhật Bias
    for (let i = 0; i < this.bias.data.length; i++) {
      const g = this.gradBias.data[i];
      this.m_b[i] = beta1 * this.m_b[i] + (1 - beta1) * g;
      this.v_b[i] = beta2 * this.v_b[i] + (1 - beta2) * (g * g);
      const m_hat = this.m_b[i] / (1 - Math.pow(beta1, t));
      const v_hat = this.v_b[i] / (1 - Math.pow(beta2, t));
      this.bias.data[i] -= (lr * m_hat) / (Math.sqrt(v_hat) + eps);
    }
  }
}

class SequentialNet {
  constructor(layers) {
    this.layers = layers;
    this.stepCount = 0;
  }

  forward(input) {
    let out = input;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }

  backward(gradOutput) {
    let grad = gradOutput;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }
    return grad;
  }

  stepAdam(lr, beta1 = 0.5) {
    this.stepCount++;
    for (const layer of this.layers) {
      layer.updateAdam(lr, beta1, 0.999, 1e-8, this.stepCount);
    }
  }
}

// Bộ tạo dữ liệu 2D mẫu cho học sinh
const DistributionGenerators = {
  heart: (numSamples) => {
    const points = [];
    for (let i = 0; i < numSamples; i++) {
      const t = Math.PI * (2 * Math.random() - 1);
      // Phương trình đường cong trái tim
      const x = 16 * Math.pow(Math.sin(t), 3) / 18;
      const y = (13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t)) / 18;
      const noise = (Math.random() - 0.5) * 0.08;
      points.push(x + noise, y + noise);
    }
    return new MiniTensor(points, [numSamples, 2]);
  },

  circle: (numSamples) => {
    const points = [];
    for (let i = 0; i < numSamples; i++) {
      const isInner = Math.random() > 0.5;
      const r = isInner ? 0.4 + (Math.random()-0.5)*0.08 : 0.85 + (Math.random()-0.5)*0.08;
      const theta = Math.random() * 2 * Math.PI;
      points.push(r * Math.cos(theta), r * Math.sin(theta));
    }
    return new MiniTensor(points, [numSamples, 2]);
  },

  spiral: (numSamples) => {
    const points = [];
    for (let i = 0; i < numSamples; i++) {
      const theta = Math.sqrt(Math.random()) * 3 * Math.PI;
      const r = theta / (3 * Math.PI) * 0.85;
      const noise = (Math.random() - 0.5) * 0.05;
      points.push((r + noise) * Math.cos(theta), (r + noise) * Math.sin(theta));
    }
    return new MiniTensor(points, [numSamples, 2]);
  },

  gaussians: (numSamples) => {
    const points = [];
    const centers = [];
    for (let k = 0; k < 8; k++) {
      const angle = (k / 8) * 2 * Math.PI;
      centers.push([0.7 * Math.cos(angle), 0.7 * Math.sin(angle)]);
    }
    for (let i = 0; i < numSamples; i++) {
      const c = centers[Math.floor(Math.random() * 8)];
      const nx = (Math.random() + Math.random() - 1) * 0.08;
      const ny = (Math.random() + Math.random() - 1) * 0.08;
      points.push(c[0] + nx, c[1] + ny);
    }
    return new MiniTensor(points, [numSamples, 2]);
  }
};

class GAN2DController {
  constructor(config = {}) {
    this.latentDim = config.latentDim || 2;
    this.hiddenDim = config.hiddenDim || 32;
    this.lrG = config.lrG || 0.005;
    this.lrD = config.lrD || 0.005;
    this.dSteps = config.dSteps || 1;
    this.distributionType = config.distribution || 'heart';
    this.batchSize = config.batchSize || 128;
    this.epoch = 0;
    this.lossHistory = { g: [], d: [] };
    this.initModels();
  }

  initModels() {
    // Generator: Latent z (2D) -> Hidden -> Hidden -> Data 2D (x, y)
    this.generator = new SequentialNet([
      new DenseLayer(this.latentDim, this.hiddenDim, 'leaky_relu'),
      new DenseLayer(this.hiddenDim, this.hiddenDim, 'leaky_relu'),
      new DenseLayer(this.hiddenDim, 2, 'tanh')
    ]);

    // Discriminator: Data (2D) -> Hidden -> Hidden -> Probability [0, 1]
    this.discriminator = new SequentialNet([
      new DenseLayer(2, this.hiddenDim, 'leaky_relu'),
      new DenseLayer(this.hiddenDim, this.hiddenDim, 'leaky_relu'),
      new DenseLayer(this.hiddenDim, 1, 'sigmoid')
    ]);

    this.epoch = 0;
    this.lossHistory = { g: [], d: [] };
  }

  trainStep() {
    const bs = this.batchSize;
    let dLossSum = 0;

    // --- BƯỚC 1: HUẤN LUYỆN DISCRIMINATOR ---
    for (let s = 0; s < this.dSteps; s++) {
      // 1.1 Dữ liệu thật
      const realData = DistributionGenerators[this.distributionType](bs);
      const dRealPred = this.discriminator.forward(realData);
      
      // BCE Loss cho ảnh thật: -log(D(x)) -> Grad: (D(x) - 1)
      const gradReal = MiniTensor.zeros(bs, 1);
      let realLoss = 0;
      for (let i = 0; i < bs; i++) {
        const p = Math.max(1e-5, Math.min(1 - 1e-5, dRealPred.data[i]));
        realLoss -= Math.log(p);
        gradReal.data[i] = (p - 1.0); // Label smoothing nhẹ
      }
      this.discriminator.backward(gradReal);

      // 1.2 Dữ liệu giả từ Generator
      const noise = MiniTensor.randn(bs, this.latentDim);
      const fakeData = this.generator.forward(noise);
      const dFakePred = this.discriminator.forward(fakeData);

      // BCE Loss cho ảnh giả: -log(1 - D(G(z))) -> Grad: D(G(z))
      const gradFake = MiniTensor.zeros(bs, 1);
      let fakeLoss = 0;
      for (let i = 0; i < bs; i++) {
        const p = Math.max(1e-5, Math.min(1 - 1e-5, dFakePred.data[i]));
        fakeLoss -= Math.log(1 - p);
        gradFake.data[i] = p;
      }
      this.discriminator.backward(gradFake);
      this.discriminator.stepAdam(this.lrD);

      dLossSum += (realLoss + fakeLoss) / (2 * bs);
    }
    const avgDLoss = dLossSum / this.dSteps;

    // --- BƯỚC 2: HUẤN LUYỆN GENERATOR ---
    // Generator muốn lừa Discriminator sao cho D(G(z)) -> 1
    const noiseG = MiniTensor.randn(bs, this.latentDim);
    const fakeDataG = this.generator.forward(noiseG);
    const dFakePredG = this.discriminator.forward(fakeDataG);

    // Non-saturating GAN Loss: -log(D(G(z))) -> Grad wrt D output: (D(G(z)) - 1)
    const gradG_D = MiniTensor.zeros(bs, 1);
    let gLoss = 0;
    for (let i = 0; i < bs; i++) {
      const p = Math.max(1e-5, Math.min(1 - 1e-5, dFakePredG.data[i]));
      gLoss -= Math.log(p);
      gradG_D.data[i] = (p - 1.0);
    }

    // Truyền ngược qua Discriminator tới Generator
    const gradToGenerator = this.discriminator.backward(gradG_D);
    this.generator.backward(gradToGenerator);
    this.generator.stepAdam(this.lrG);

    const avgGLoss = gLoss / bs;
    this.epoch++;

    this.lossHistory.d.push(avgDLoss);
    this.lossHistory.g.push(avgGLoss);
    if (this.lossHistory.d.length > 80) {
      this.lossHistory.d.shift();
      this.lossHistory.g.shift();
    }

    return {
      epoch: this.epoch,
      dLoss: avgDLoss,
      gLoss: avgGLoss
    };
  }

  sampleGenerator(numSamples = 200) {
    const noise = MiniTensor.randn(numSamples, this.latentDim);
    const fakePoints = this.generator.forward(noise);
    const res = [];
    for (let i = 0; i < numSamples; i++) {
      res.push([fakePoints.data[i * 2], fakePoints.data[i * 2 + 1]]);
    }
    return res;
  }

  evaluateGrid(gridSize = 25) {
    // Tính xác suất phân loại của Discriminator trên lưới 2D [-1.2, 1.2]
    const gridPoints = [];
    const step = 2.4 / gridSize;
    for (let i = 0; i < gridSize; i++) {
      const y = 1.2 - i * step;
      for (let j = 0; j < gridSize; j++) {
        const x = -1.2 + j * step;
        gridPoints.push(x, y);
      }
    }
    const input = new MiniTensor(gridPoints, [gridSize * gridSize, 2]);
    const preds = this.discriminator.forward(input);
    return {
      gridSize,
      probs: preds.data
    };
  }
}

if (typeof module !== 'undefined') {
  module.exports = { GAN2DController, DistributionGenerators, MiniTensor };
}
