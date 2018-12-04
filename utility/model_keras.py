from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, nadam, rmsprop
from keras.applications import Xception, ResNet50, InceptionV3
from keras import regularizers

# Parameters
# H             : Height of input image
# W             : Width of input image
# n_class       : Number of output class
# learning_rate : Learning rate
def model(H, W, n_class, learning_rate, decay):
    X_input = Input(shape = (H, W, 3))

    # use pretrained imagenet model
    xception_model = ResNet50(weights='imagenet', include_top=False)
    for i, layers in enumerate(xception_model.layers):
        if i < 126:
            layers.trainable = False
        else:
            layers.trainable = False

    x = xception_model(X_input)
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    #x = Dense(512, activation='relu', kernel_regularizer = regularizers.l2(0.01))(x)
    #x = Dropout(0.5)(x)
    #x = BatchNormalization()(x)
    #x = Dense(32, activation='relu', kernel_regularizer = regularizers.l2(0.01))(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.3)(x)

    if (n_class == 2):
        predictions = Dense(n_class-1, activation='sigmoid')(x)
    else:
        predictions = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=X_input, outputs=predictions)

    # print model summary
    model.summary()

    if (n_class == 2):
        model.compile(optimizer=nadam(lr=learning_rate), loss='binary_crossentropy',metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(lr=learning_rate, decay=decay), loss='categorical_crossentropy',metrics=['accuracy'])

    return model
