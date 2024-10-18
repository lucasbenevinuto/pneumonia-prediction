# Identificação de Pneumonia em Raios-X de Pulmão

## Descrição do Projeto

Este projeto tem como objetivo desenvolver um sistema de visão computacional capaz de identificar pneumonia em imagens de raios-X de pulmão. O projeto utiliza o TensorFlow e implementa dois modelos: um modelo próprio desenvolvido durante o projeto e o modelo VGG16, que é uma rede neural convolucional pré-treinada.

## Tecnologias Utilizadas

- Python
- TensorFlow
- Keras
- NumPy

## Estrutura do Projeto

```
projeto_pneumonia/
│
├── train/                     # Pasta contendo as imagens de raios-X
│   ├── pneumonia/               # Imagens de pacientes com pneumonia
│   └── normal/                  # Imagens de pacientes saudáveis
├── test/                     # Pasta contendo as imagens de raios-X
│
│   ├── pneumonia/               # Imagens de pacientes com pneumonia
│   └── normal/
│                        # Código-fonte do projeto
├── modelo_pneumonia.py        # Implementação do modelo próprio
├── modelo_pneumonia_vgg16.py           # Implementação do modelo VGG16
│
├── README.md                    # Este arquivo
└── teste.py                      # Script principal para execução
```

## Como Executar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu_usuario/pneumonia_prediction.git
   cd projeto_pneumonia
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare o Dataset:**
   - Coloque suas imagens de raios-X nas pastas `dataset/pneumonia` e `dataset/normal`.

4. **Treine o Modelo:**
   - Para treinar o modelo próprio:
     ```bash
     python src/modelo_proprio.py
     ```
   - Para treinar o modelo VGG16:
     ```bash
     python src/vgg16_model.py
     ```

5. **Execute a Predição:**
   - Execute o script principal para classificar uma nova imagem de raio-X:
     ```bash
     python main.py --image caminho/para/imagem.jpg
     ```

## Resultados

Os resultados do modelo serão exibidos na tela, indicando se a imagem contém pneumonia ou se está normal.
