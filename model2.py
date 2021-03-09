import tensorflow as tf

class mynet(tf.keras.Model):

  def __init__(self):
    super(mynet, self).__init__()

    self.l1_conv    = tf.keras.layers.Conv2D(filters=32,kernel_size=15,strides=1,activation='relu',padding="valid", kernel_initializer=tf.keras.initializers.he_normal(),input_shape=(100,100,3))
    self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
    self.l1_batch   = tf.keras.layers.BatchNormalization()

    self.l2_conv    = tf.keras.layers.Conv2D(filters=64,kernel_size=8,strides=1,activation='relu',padding="valid", kernel_initializer=tf.keras.initializers.he_normal(),input_shape=(100,100,3))
    self.l2_max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=3)
    self.l2_batch   = tf.keras.layers.BatchNormalization()

    self.l3_conv    = tf.keras.layers.Conv2D(filters=256,kernel_size=5,strides=1,activation='relu',padding="valid", kernel_initializer=tf.keras.initializers.he_normal(),input_shape=(100,100,3))
    self.l3_max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
    self.l3_batch   = tf.keras.layers.BatchNormalization()
    
    self.fc1 = tf.keras.layers.Flatten()
    self.fc2   = tf.keras.layers.Dense(64, activation='relu')

  
  def call(self, x):
    x = self.l1_conv(x)
    x = self.l1_max_pool(x)
    x = self.l1_batch(x)
    x = self.l2_conv(x)
    x = self.l2_max_pool(x)
    x = self.l2_batch(x)
    x = self.l3_conv(x)
    x = self.l3_max_pool(x)
    x = self.l3_batch(x)
    x = self.fc1(x)
    x = self.fc2(x)   

    return x

def contrastive_loss(model1, model2, y, margin):
    d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keepdims=True))
    tmp = y * tf.square(d)    
    tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_mean(tmp + tmp2)/2