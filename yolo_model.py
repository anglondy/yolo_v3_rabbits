from constants import tf


# Class to create, compile and train the model
class YoloV3:
    def __init__(self, num_classes: int = 20, anchors: int = 3, train_vgg: bool = False):
        self.anchors = anchors
        self.num_classes = num_classes
        self.train_vgg = train_vgg
        self.model = self.get_model()

    @staticmethod
    def get_dbl(x, n_filters=256):  # Conv -> Batch Norm -> Leaky Relu block
        conv = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(x)
        batch = tf.keras.layers.BatchNormalization()(conv)
        return tf.keras.layers.LeakyReLU(alpha=0.1)(batch)

    def get_model(self):
        vgg = tf.keras.applications.VGG19(include_top=False, input_shape=(224, 224, 3))
        if not self.train_vgg:
            vgg.trainable = False

        x = vgg.output
        for i in range(5):
            x = self.get_dbl(x)

        # First scale output with shape (n, 7, 7, anchors * (5 + num_classes))
        pre_output1 = self.get_dbl(x, n_filters=1024)
        output1 = tf.keras.layers.Conv2D((self.anchors * (5 + self.num_classes)), 3, padding='same')(pre_output1)

        # Taking from model layer with shape (n, 14, 14, 512)
        vgg_14 = vgg.get_layer('block5_conv4').output

        x = self.get_dbl(x, n_filters=512)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, vgg_14])

        for i in range(5):
            x = self.get_dbl(x)

        # Second scale output with shape (n, 14, 14, anchors * (5 + num_classes))
        pre_output2 = self.get_dbl(x, n_filters=512)
        output2 = tf.keras.layers.Conv2D((self.anchors * (5 + self.num_classes)), 3, padding='same')(pre_output2)

        # Taking from model layer with shape (n, 28, 28, 256)
        vgg_28 = vgg.get_layer('block4_conv4').output

        x = self.get_dbl(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, vgg_28])

        for i in range(5):
            x = self.get_dbl(x)

        # Last scale output with shape (n, 28, 28, anchors * (5 + num_classes))
        pre_output3 = self.get_dbl(x)
        output3 = tf.keras.layers.Conv2D((self.anchors * (5 + self.num_classes)), 3, padding='same')(pre_output3)

        # Final model
        model = tf.keras.Model(inputs=vgg.input, outputs=[output1, output2, output3])
        return model

    def train(self, x_train, y_train, epochs=10, **kwargs):
        return self.model.fit(x_train, y_train, epochs=epochs, **kwargs)  # returns history object

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def summary(self):
        return self.model.summary()

    def compile(self, loss, lr: float = 1e-4):
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    def save(self, path: str) -> None:
        self.model.save(path)
