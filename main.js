// Configuration
var EPOCHS = 10;

// Variables
var RAW_DATA = null;
var WEIGHTS = null;
var DATA = null;
var UI = null;

function preload() {
  console.log("👉 Preload");
  RAW_DATA = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");
  prepareData();
  validateModel();
}

function predict(inputs) {
  return inputs.dot(WEIGHTS);
}

function draw() {
  
}

function prepareData() {
  console.log("👉 prepareData")
  DATA = new Data(RAW_DATA);
  console.log(DATA.training);
}

function loss(predicted, actual) {
  return predicted // So e.g. [1,1,1] - [1,0,0]
    .sub(actual) // should result in [0,1,1]
    .square()
    .mean();
}

function acc(predicted, actual) {
  return tf
    .sign(predicted)
    .mul(actual)
    .mean();
}

function validateModel() {
  console.log("👉 validateModel");

  // This should be the IDEAL set of target weights!
  WEIGHTS = tf
    .variable(tf.tensor([[1, 1, 1, 0, 0, 0, -1, -1, -1]]))
    .transpose();

  // Small set of inputs
  const inputs = tf.tensor([
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0.5, 0, 0.3, 0, 0, 0.7, 0]
  ]);
  // Small set of labels, 1 means top 3 is higher, -1 means bottom 3 is higher
  const labels = tf.tensor([[1, -1, -1]]).transpose();

  console.log("PREDICTED LABELS -->");
  predict(inputs).print();
  console.log("ACTUAL LABELS -->");
  labels.print();
  console.log("LOSS -->");
  // For this to work I think the loss has to be 0 here, instead it's 0.2903703451156616
  loss(predict(inputs), labels).print();
  console.log("ACC -->");
  // For this to work I think the loss has to be 0 here, instead it's 0.2903703451156616
  acc(predict(inputs), labels).print();
}
  
