# Neural-Network-Systems

<p align="center">
  <img src="https://github.com/VsIG-official/Neural-Network-Systems/blob/master/docs/NNS.png" data-canonical-src="https://github.com/VsIG-official/Neural-Network-Systems/blob/master/docs/NNS.png" width="600" height="300" />
</p>

## Table of Contents

- [Description](#description)
- [Badges](#badges)
- [Contributing](#contributing)
- [License](#license)

### Description

This repo contains all of My work for Neural Network Systems

## Badges

![Theme](https://img.shields.io/badge/Theme-Neural--Networks-blueviolet.svg?style=flat-square)
![Language](https://img.shields.io/badge/Language-CSharp-blueviolet.svg?style=flat-square)
![Language](https://img.shields.io/badge/Language-Python-green.svg?style=flat-square)


---

## Example

```python
model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size, batch_size),
                           tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 64)),
                           tf.keras.layers.Dense(units = 64, activation = activation_type),
                           tf.keras.layers.Dense(units = 1)
])

# BinaryCrossEntropy = two label classes
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = [metrics_type])
```

---

## Contributing

> To get started...

### Step 1

- 🍴 Fork this repo!

### Step 2

- **HACK AWAY!** 🔨🔨🔨

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2022 © <a href="https://github.com/VsIG-official" target="_blank">VsIG</a>.
