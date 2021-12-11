const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node');

function normalize(tensor) {
  const min = tensor.min();
  const max = tensor.max();
  return tensor.sub(min).div(max.sub(min));
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'linear',
    inputDim: 1,
  }));

  const optimizer = tf.train.sgd(0.1);
  model.compile({
    loss: 'meanSquaredError',
    optimizer,
  })

  return model;
}

function trainModel(model, trainingFeatureTensor, trainingLabelTensor) {
  return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    batchSize: 32,
    epochs: 20,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    }
  });
}

async function main() {
  // Load data from CSV
  const usedCarPricesDataset = tf.data.csv("file://Used_Car_Price_Data.csv");
  const pointsDataset = usedCarPricesDataset.map(el => ({
    x: el.Milage,
    y: el['Price($)'],
  }));
  const points = await pointsDataset.toArray();

  // Remove one element if length is odd
  if (points.length % 2 !== 0) {
    points.pop();
  }

  tf.util.shuffle(points);

  // Extract features (inputs)
  const featureValues = points.map(el => typeof el.x === "string" ? parseFloat(el.x.replace(/,/g, '')) : el.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  // Extract labels (outputs)
  const labelValues = points.map(el => typeof el.y === "string" ? parseFloat(el.y.replace(/,/g, '')) : el.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  // Normalize
  const normalizedFeature = normalize(featureTensor);
  const normalizedLabel = normalize(labelTensor);

  const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeature, 2);
  const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabel, 2);

  const model = createModel();
  const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
  console.log(`Training set loss: ${result.history.loss.pop()}`);
  console.log(`Validation set loss: ${result.history.val_loss.pop()}`);

  const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
  const loss = await lossTensor.dataSync();
  console.log(`Testing set loss: ${loss}`);
}

main().then(() => console.log('Done!'));
