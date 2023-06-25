from tensorflow.keras.callbacks import EarlyStopping


def EarlyStopping_callback():
    return EarlyStopping(monitor='val_loss',
                        patience=3,
                        min_delta=0.01,
                        verbose=1,
                        mode='auto',
                        restore_best_weights=True)