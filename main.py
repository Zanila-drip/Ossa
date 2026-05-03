
#Detectar imagenes
#Analizar imagenes
#Reglas de inferencia
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

trainData = tf.keras.utils.image_dataset_from_directory(
    "crop_images",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)