import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo salvo
model = load_model('modelo_pneumonia_vgg16.h5')


# Caminho para a pasta contendo as imagens
test_dir = 'C:/Users/lucao/OneDrive/Documents/Portifolio/Projetos/Computer Vision/Chest/chest_xray/test/PNEUMONIA'

# Listar todos os arquivos na pasta
images = os.listdir(test_dir)

positivo = 0
negativo = 0

# Percorrer todas as imagens da pasta
for img_name in images:
    img_path = os.path.join(test_dir, img_name)
    
    # Carregar e processar a imagem
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Fazer a predição
    prediction = model.predict(img_array)

    # Exibir o resultado
    if prediction[0] > 0.5:
        print(f"{img_name}: Pneumonia detectada")
        positivo += 1
    else:
        print(f"{img_name}: Raio-X normal")
        negativo += 1


total = positivo + negativo
p_positivo = (positivo*100)/total
p_negativo = (negativo*100)/total

print(f'positivo: {positivo}/ Porcentagem: {p_positivo:.2f}')
print(f'negativo: {negativo}/ Porcentagem: {p_negativo:.2f}')