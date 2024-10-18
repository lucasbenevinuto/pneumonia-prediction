import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Caminho do dataset
dataset_dir = 'C:/Users/lucao/OneDrive/Documents/Portifolio/Projetos/Computer Vision/Chest/chest_xray/train'

# Criar geradores de dados com Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Ajuste de brilho
    shear_range=0.2                # Cisalhamento
)


# Gerador de dados de treinamento
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),       # Tamanho das imagens
    batch_size=32,
    class_mode='binary',          # Saída binária (normal vs pneumonia)
    subset='training'             # Dados de treinamento
)

# Gerador de dados de validação
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',          # Saída binária (normal vs pneumonia)
    subset='validation'           # Dados de validação
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Camada adicional
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# Avaliar o modelo no conjunto de validação
# Avaliar o modelo no conjunto de validação
results = model.evaluate(validation_generator)
loss, accuracy, precision, recall = results
print(f"Acurácia: {accuracy*100:.2f}%, Precisão: {precision*100:.2f}%, Recall: {recall*100:.2f}%")

# Prever os dados de validação
y_pred = model.predict(validation_generator)
y_pred_classes = (y_pred > 0.3).astype(int)  # Ajuste o limiar aqui

# Para avaliar o desempenho com o novo limiar
from sklearn.metrics import classification_report

# Obter as classes verdadeiras
y_true = validation_generator.classes

# Relatório de classificação
print(classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys()))

# Salvar o modelo treinado
model.save('modelo_pneumonia.h5')
