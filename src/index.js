const express = require('express');
const multer = require('multer');
const fs = require('fs');
const { promisify } = require('util');
const { loadImage } = require('@tensorflow/tfjs-node');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

const upload = multer({ dest: 'static/uploads/' });

const MODEL_FILE = 'soil_types_model.h5';
const LABELS_FILE = 'labels.txt';

let model;
let labels;

const loadModelAndLabels = async () => {
  try {
    model = await tf.loadLayersModel(`file://${MODEL_FILE}`);
    labels = fs.readFileSync(`${MODEL_DIRECTORY}/${LABELS_FILE}`, 'utf-8').split('\n');
  } catch (error) {
    console.error('Error loading model and labels:', error);
    throw error;
  }
};

// Panggil fungsi loadModelAndLabels saat aplikasi dimulai
loadModelAndLabels().catch((error) => {
  console.error('Error loading model and labels:', error);
  process.exit(1);
});

const predictSoilType = async (imagePath) => {
  const img = await loadImage(imagePath);
  const resizedImg = tf.image.resizeBilinear(img, [224, 224]);
  const expandedImg = resizedImg.expandDims(0);
  const normalizedImageArray = expandedImg.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
  const predictions = model.predict(normalizedImageArray);
  const index = predictions.argMax(1).dataSync()[0];
  const className = labels[index];
  const confidenceScore = predictions.dataSync()[index];
  return { className: className.substring(2), confidenceScore };
};

app.get('/', (req, res) => {
  res.send('Hello Indonesia!');
});

app.post('/prediction', upload.single('image'), async (req, res) => {
  try {
    const { file } = req;
    const imagePath = file.path;
    
    const { className, confidenceScore } = await predictSoilType(imagePath);

    // Hapus gambar lokal setelah diunggah
    // await promisify(fs.unlink)(imagePath);

    return res.json({
      status: { code: 200, message: 'Success predicting' },
      data: { soil_types_prediction: className, confidence: confidenceScore },
    });
  } catch (error) {
    return res.status(500).json({
      status: { code: 500, message: 'Internal Server Error' },
      data: null,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});