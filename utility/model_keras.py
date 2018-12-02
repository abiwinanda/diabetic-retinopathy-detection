from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D
from keras.models import Model
from keras.optimizers import Adam, nadam
from keras.applications.xception import Xception
from keras import regularizers

# Parameters
# H             : Height of input image
# W             : Width of input image
# n_class       : Number of output class
# learning_rate : Learning rate
def model(H, W, n_class, learning_rate, decay):
    X_input = Input(shape = (H, W, 3))

    # use pretrained imagenet model
    xception_model = Xception(weights='imagenet', include_top=False)
    for i, layers in enumerate(xception_model.layers):
        if i <= 15:
            layers.trainable = True
        else:
            layers.trainable = True

    x = xception_model(X_input)
    x = GlobalAveragePooling2D()(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    if (n_class == 2):
        predictions = Dense(n_class-1, activation='sigmoid')(x)
    else:
        predictions = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=X_input, outputs=predictions)

    # print model summary
    model.summary()

    if (n_class == 2):
        model.compile(optimizer=Adam(lr=learning_rate, decay=decay), loss='binary_crossentropy',metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(lr=learning_rate, decay=decay), loss='categorical_crossentropy',metrics=['accuracy'])

    return model
