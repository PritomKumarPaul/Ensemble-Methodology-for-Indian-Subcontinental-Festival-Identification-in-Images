
from keras.models import Sequential

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(tf.test.gpu_device_name())
"""
from keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  
set_session(tf.compat.v1.Session(config=config))
"""
from tensorflow.python.keras.layers import Dense, Flatten
#from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# data_dir="F:/soft-computing-project-master/festivals-Splitted/train"
data_dir="F:/AUST/4.2/8. CSE4238 - Soft Computing Lab/Project/festivals dataset/festivals-Splitted/train"
img_height,img_width=224,224
batch_size=64
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)
test_dir="F:/AUST/4.2/8. CSE4238 - Soft Computing Lab/Project/festivals dataset/festivals-Splitted/test"

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

x=len(train_ds.class_names)

model = Sequential()



model.add(tf.keras.layers.InputLayer(input_shape=(img_height,img_width, 3), dtype=tf.uint8))
model.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)))
model.add(tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input))


pretrained_model= tf.keras.applications.EfficientNetB7(include_top=False,
                   input_shape=(img_height,img_width,3),
                   pooling='avg',
                   weights='imagenet')
for layer in pretrained_model.layers[:int(len(pretrained_model.layers)*.75)]:   
       layer.trainable=False

model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(x, activation='softmax'))

#[:int(len(pretrained_model.layers)*.75)]:
"""
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=50000,
    decay_rate=0.9)
"""

checkpoint_filepath = 'F:/AUST/4.2/8. CSE4238 - Soft Computing Lab/Project/ckpt.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)



model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
epochs=15





from datetime import datetime
start = datetime.now()
history = model.fit(
  train_ds,
  validation_data=val_ds,
  verbose="auto",
  callbacks=[model_checkpoint_callback],
  epochs=epochs
)
duration = datetime.now() - start
print("Training completed in time: ", duration)


result=model.evaluate(
    test_ds,
    batch_size=batch_size,
    verbose="auto"
)

"""

save

model.save_weights('F:/AUST/4.2/8. CSE4238 - Soft Computing Lab/Project/efficientnetb7.h5', overwrite=True, save_format=None, options=None)


"""


"""

load

model.load_weights('F:/AUST/4.2/8. CSE4238 - Soft Computing Lab/Project/ckpt.h5', skip_mismatch=False, by_name=False, options=None)

model.load_weights('F:/AUST/4.2/8. CSE4238 - Soft Computing Lab/Project/efficientnetb7.h5', skip_mismatch=False, by_name=False, options=None)


"""
