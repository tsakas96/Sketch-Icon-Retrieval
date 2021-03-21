import tensorflow as tf

class mynet(tf.keras.Model):

  def __init__(self, NUM_CLASSES=0):
    super(mynet, self).__init__()

    self.l1_conv    = tf.keras.layers.Conv2D(filters=32,kernel_size=15,strides=1,activation='relu',padding="valid", kernel_initializer=tf.keras.initializers.glorot_uniform()) #,input_shape=(100,100,3)
    self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
    self.l1_batch   = tf.keras.layers.BatchNormalization()

    self.l2_conv    = tf.keras.layers.Conv2D(filters=64,kernel_size=8,strides=1,activation='relu',padding="valid", kernel_initializer=tf.keras.initializers.glorot_uniform()) #,input_shape=(43,43,32)
    self.l2_max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=3)
    self.l2_batch   = tf.keras.layers.BatchNormalization()

    self.l3_conv    = tf.keras.layers.Conv2D(filters=256,kernel_size=5,strides=1,activation='relu',padding="valid", kernel_initializer=tf.keras.initializers.glorot_uniform()) #,input_shape=(12,12,64)
    self.l3_max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
    self.l3_batch   = tf.keras.layers.BatchNormalization()
    
    self.fc1 = tf.keras.layers.Flatten()
    self.fc2   = tf.keras.layers.Dense(64, activation='relu')
    self.classification_outpout = tf.keras.layers.Dense(NUM_CLASSES, activation='linear')

  
  def call(self, x, training = None):
    x = self.l1_conv(x)
    x = self.l1_max_pool(x)
    x = self.l1_batch(x, training = training)
    x = self.l2_conv(x)
    x = self.l2_max_pool(x)
    x = self.l2_batch(x, training = training)
    x = self.l3_conv(x)
    x = self.l3_max_pool(x)
    x = self.l3_batch(x, training = training)
    x = self.fc1(x)
    features = self.fc2(x)
    class_output = self.classification_outpout(features)
    class_output = tf.nn.softmax(class_output, axis = 1)

    return features, class_output


def contrastive_loss(model1, model2, y, margin):
  d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keepdims=True))
  tmp = y * tf.square(d)    
  tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
  return tf.reduce_mean(tmp + tmp2)/2

def siamese_loss(model1, model2, y):
  Q = 10
  alpha = 2/Q
  beta = 2*Q
  gamma = -2.77/Q
  d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keepdims=True))
  loss = y * alpha * tf.square(d) + (1-y) * beta * tf.exp(gamma*d)
  return loss

def triplet_loss(sketches, positive_icons, negative_icons, margin):

	positive_dist = tf.reduce_sum(tf.square(tf.subtract(sketches, positive_icons)), -1)
	negative_dist = tf.reduce_sum(tf.square(tf.subtract(sketches, negative_icons)), -1)

	loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), margin)
	loss = tf.reduce_sum(tf.maximum(loss_1, 0.0))

	return loss

def compute_cross_entropy(logits, labels):
  cross_entropy= -tf.reduce_mean(tf.reduce_sum(labels*tf.math.log(tf.clip_by_value(logits,1e-10,1.0)), axis = 1))
  return cross_entropy