// Define the model
const model = tf.sequential();
model.add(
  tf.layers.conv2d({
    inputShape: [32, 32, 3],
    filters: 32,
    kernelSize: 3,
    activation: "relu",
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 64, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

// Dummy train function (replace with real data!)
async function train() {
  const xs = tf.randomNormal([100, 32, 32, 3]); // Simulated input
  const ys = tf.oneHot(tf.randomUniform([100], 0, 10, "int32"), 10); // Simulated labels
  await model.fit(xs, ys, {
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "acc"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}
