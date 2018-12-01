from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D
from keras.models import Model
from keras.optimizers import Adam, nadam
from keras.applications.xception import Xception
from keras import regularizers

def model(H, W, n_class, learning_rate):
    X_input = Input(shape = (H, W, 3))
    
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
    
    predictions = Dense(n_class-1, activation='sigmoid')(x)

    model = Model(inputs=X_input, outputs=predictions)
    model.summary()
    model.compile(optimizer=Adam(lr=learning_rate, decay=3e-4), loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model