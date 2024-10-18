import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import numpy as np
from sklearn.metrics import classification_report

# Caminho do dataset
dataset_dir = 'C:/Users/lucao/OneDrive/Documents/Portifolio/Projetos/Computer Vision/Chest/chest_xray/train'

# Criar geradores de dados com Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,               # Normalização dos pixels
    validation_split=0.2,         # Separar 20% para validação
    rotation_range=20,            # Rotação aleatória das imagens
    width_shift_range=0.2,        # Deslocamento horizontal
    height_shift_range=0.2,       # Deslocamento vertical
    zoom_range=0.2,               # Aplicar zoom
    horizontal_flip=True          # Inversão horizontal
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

# Carregar o modelo VGG16 pré-treinado sem as camadas de classificação
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Congelar as camadas do modelo base
base_model.trainable = False

# Adicionar camadas personalizadas ao topo do modelo
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),   # Dropout para evitar overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária (pneumonia ou não)
])

# Compilar o modelo com recall e precision como métricas adicionais
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Avaliar o modelo no conjunto de validação
loss, accuracy, recall, precision = model.evaluate(validation_generator)
print(f"Acurácia: {accuracy*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"Precisão: {precision*100:.2f}%")

# Prever probabilidades no conjunto de validação
y_pred_prob = model.predict(validation_generator)
y_pred = (y_pred_prob > 0.3).astype(int)  # Ajustando o limiar para 0.3

# Obter as verdadeiras classes
y_true = validation_generator.classes

# Gerar relatório de classificação
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

# Salvar o modelo treinado
model.save('modelo_pneumonia_vgg16.h5')
