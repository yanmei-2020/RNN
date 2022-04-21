from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words = 500)
# 어휘사전에 500개 단어만 들어 있음
# 25000개의 구절/샘플로 이루어 짐
#print(train_input.shape, test_input.shape)
# 리뷰를 1열로 1차원 데이터가 되게 만듬
# train_input = [리뷰1, 리뷰2...]
# 이미 토큰화 되어있음
#print(train_input[0])
# 부정 0과 긍정 1로 나뉨
#print(train_target[:20])

from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state=42)

import numpy as np
lengths = [len(x) for x in train_input]
lenghts = np.array(lengths)

import matplotlib.pyplot as plt
#plt.hist(lengths)
#plt.xlabel('length')
#plt.ylabel('frequency')
#plt.show()

# 리뷰의 단어를 100개로 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape)
val_seq = pad_sequences(val_input, maxlen=100)

from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
# 샘플의 길이가 100이므로 차원이 100임
# 인풋을 원핫 인코딩으로 표현하므로 0~499까지의 숫자를 500길이의 배열로 표현할 수 있다.
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)
model.summary()

# Training
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('bes-simplernn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
validation_data=(val_oh, val_target),
callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legent(['train', 'val'])
plt.show()