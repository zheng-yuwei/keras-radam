# Keras RAdam

\[[中文](https://github.com/zheng-yuwei/keras-radam/blob/master/README.zh-CN.md)|[English](https://github.com/zheng-yuwei/keras-radam/blob/master/README.md)\]

基于tensorflow.keras框架的[RAdam](https://arxiv.org/abs/1908.03265)非官方实现

## 使用

```python
import numpy as np
from tensorflow import keras
from radam import RAdam

# Build toy model with RAdam optimizer
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape=(17,), units=3))
model.compile(RAdam(), loss='mse')

# Generate toy data
x = np.random.standard_normal((4096 * 30, 17))
w = np.random.standard_normal((17, 3))
y = np.dot(x, w)

# Fit
model.fit(x, y, epochs=5)
```

用keras.callbacks设置动态学习率：
```python
import numpy as np
from tensorflow import keras
from radam import RAdam

# Build toy model with RAdam optimizer
model = keras.models.Sequential()
model.add(keras.layers.Dense(input_shape=(17,), units=3))
model.compile(RAdam(), loss='mse')

# Generate toy data
x = np.random.standard_normal((4096 * 30, 17))
w = np.random.standard_normal((17, 3))
y = np.dot(x, w)

# Fit
lr_callback = keras.callbacks.LearningRateScheduler(lambda cur_epoch: 1e-3 if cur_epoch < 10 else 1e-4)
model.fit(x, y, epochs=5, callbacks=[lr_callback])
```

## Warmup阶段的学习率调整
```python
from radam import RAdam

# Warmup阶段的学习率： lr = lr * warmup_coef
RAdam(lr=1e-3, warmup_coef=0.1)
```

## 参考
[CyberZHG/keras-radam](https://github.com/CyberZHG/keras-radam)
