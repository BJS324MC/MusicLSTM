class Attention extends tf.layers.Layer {
  constructor(config = { returnSequence: true }) {
    super(config);
    this.returnSequences = config.returnSequences;
  }
  build(inputShape) {
    this.w = this.addWeight("w", [inputShape[inputShape.length - 1], 1], "float32", tf.initializers.ones());
    this.b = this.addWeight("b", [inputShape[1], 1], "float32", tf.initializers.zeros());
  }
  call(x) {
    let e = tf.tanh(tf.dot(x, this.w) + this.b),
      a = tf.softmax(e, 1),
      output = x * a;
    if (this.returnSequences)
      return output;
    return tf.sum(output, 1);
  }
  getConfig() {
    const config = super.getConfig();
    Object.assign(config, { returnSequences: this.returnSequences });
    return config;
  }
  static get className() {
    return "Attention";
  }
}
tf.serialization.registerClass(Attention);

function createModel(inputLength = 100, outputLength = 40) {
  const model = tf.sequential();
  model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units: inputLength, returnSequences: true }), inputShape: [inputLength, 1] }));
  model.add(new Attention());
  model.add(tf.layers.lstm({ units: inputLength }));
  model.add(tf.layers.dense({ units: Math.floor(inputLength / 2) }));
  model.add(tf.layers.dropout({rate:0.3}));
  model.add(tf.layers.dense({ units: outputLength, activation: "softmax" }));
  return model;
}
async function lstm() {
  return await createModel();
}
async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: "rmsprop",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
  const batchSize = 32,
    epochs = 50;
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks({ name: "Training Performance" },
      ["loss", "categoricalCrossentropy"], { height: 200, callbacks: ["onEpochEnd"] }
    )
  });
}
const midi = new Midi();
async function loadData() {
  return await Promise.all([
    //Midi.fromUrl("http://localhost:7700/data/0.midi"),
    //Midi.fromUrl("http://localhost:7700/data/1.midi"),
    //Midi.fromUrl("http://localhost:7700/data/2.midi"),
    //Midi.fromUrl("http://localhost:7700/data/3.midi"),
    Midi.fromUrl("http://localhost:7700/data/4.mid")
  ]);
}

function playRawData(data){
  data.tracks[0].notes.forEach(y => {
    sampler.triggerAttackRelease(y.name,y.duration,y.time);
  })
}
function cleanData(data) {
  let d = Array.from({ length: data.length }, () => []),
    timelines = Array.from({ length: data.length }, () => ({}));
  data.forEach((a, i) => a.tracks[0].notes.forEach(y => {
    let time = Math.round(y.time * 100000) / 100000,
      stop = time + Math.round(y.duration * 100000) / 100000,
      note = y.name,
      timeline = timelines[i];
    if (time in timeline) timeline[time].push("s" + note);
    else timeline[time] = ["s" + note];
    if (stop in timeline) timeline[stop].push("e" + note);
    else timeline[stop] = ["e" + note];
  }));
  timelines.forEach((timeline, i) => {
    let n = 0;
    Object.keys(timeline).sort((a, b) => (+a) - (+b)).forEach(a => {
      if (a - n) d[i].push("t" + (a - n));
      d[i].push(...timeline[a]);
      n = a;
    })
  });
  console.log(JSON.stringify(data))
  return d;
}

function createTable(data) {
  let table = {},
    i = 0;
  data.forEach(a => a.forEach(b => (b in table) ? 0 : table[b] = i++));
  Object.defineProperty(table, "length", {
    value: i
  });
  return table
}

const encodeData = (data, table) => data.map(a=>a.map(b => table[b]/table.length)),
      oneHot = (index, length) => {
        let arr=Array.from({length:length},()=>0);
        arr[index]=1;
        return arr;
      },
      shuffleCombo = (a,b) => {
        let permu=Array.from({length:a.length},(a,i)=>i);
        for(let i=permu.length-1;i>=0;i--){
          let j=Math.floor(Math.random()*permu.length);
          [permu[i],permu[j]]=[permu[j],permu[i]];
        }
        return [a.map((e,i)=>a[permu[i]]),b.map((e,i)=>b[permu[i]])]
      }
function trainingData(data,table,sequenceLength=2){
  let inputs=[],
      outputs=[];
  data=encodeData(data,table);
  for(let a in data)for(let i=0;i<data[a].length-sequenceLength;i++){
    let p=i+sequenceLength,
        d=data[a];
    inputs.push(d.slice(i,p))
    outputs.push(oneHot(d[p],table.length));
  }
  return shuffleCombo(inputs,outputs);
}

function playData(data) {
  let stack = [],
    time = 0,
    now = Tone.now();
  data.forEach(a =>
    a[0] === "s" ? stack[a.slice(1)] = time :
    a[0] === "t" ? time += Number(a.slice(1)) :
    sampler.triggerAttackRelease(a.slice(1), time - stack[a.slice(1)], now + stack[a.slice(1)]));
}

/* Finishing the network */

let net, p, urls = {},
  P = ["C", "D", "E", "F", "G", "A", "B"],
  V = ["C", "E", "G", "A"],
  data = loadData();
data.then(a => data = a);

for (let i = 0; i < 7; i++) P.forEach(a => {
  urls[a + i] = a + i + ".wav";
  urls[a + "#" + i] = a + "s" + i + ".wav"
});

const sampler = new Tone.Sampler({
  urls: urls,
  baseUrl: "http://localhost:7700/instruments/piano/"
}).toDestination();

Tone.loaded().then(() => {
  playRawData(data[0]);
  //const loop = new Tone.Loop(()=>playData(data[1]),"3m").start(0);
  //Tone.Transport.start();
  //playData(data[0])
});
/*addEventListener("load", () => {
  lstm().then(model => {
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);
    //p=model.save("http://localhost:7700/models");
    //p.then(a=>p=a);
    /*let table=createTable(data),
        trainingSet=trainingData(data,table,50);
    trainModel(model,trainingSet[0],trainingSet[1])
    .then(a=>{net=a;await model.save("http://localhost:7700/models/model");})
  });
});*/