# **DeepSeek con Ollama**

![Python](https://img.shields.io/badge/Python-3.7+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Latest-00A67E.svg?style=for-the-badge)
![Mistral](https://img.shields.io/badge/Mistral-7B-FF6B6B.svg?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-4.36+-FFD43B.svg?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-FFD43B.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg?style=for-the-badge)
![GPU](https://img.shields.io/badge/GPU-Supported-00A67E.svg?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Custom-FF6B6B.svg?style=for-the-badge)

Este proyecto integra el modelo DeepSeek en Ollama, permitiendo su carga y ejecución de manera sencilla. A continuación, se detallan los pasos para instalar las dependencias, configurar y ejecutar el modelo.

## **Requisitos**

## Tecnologías

- Python 3.7+
- Ollama
- Mistral-7B
- Transformers 4.36.0
- PyTorch 2.1.0
- Hugging Face Hub
- CUDA (opcional para GPU)

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/hectorflores28/python-ollama-deepseek.git
cd python-ollama-deepseek
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar Hugging Face:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

5. Instalar Ollama:
- Descargar e instalar desde [ollama.ai/download](https://ollama.ai/download)
- Verificar instalación: `ollama --version`

## Uso

1. Crear el modelo en Ollama:
```bash
ollama create velas-assistant -f ollama.modelfile
```

2. Entrenar el modelo:
```bash
python Trainning.py
```

3. Guardar en Hugging Face:
```bash
python SaveHuggingFace.py
```

4. Ejecutar el modelo:
```bash
ollama run velas-assistant
```

## Dataset

El proyecto incluye un dataset personalizado con:
- Conversaciones de ejemplo
- Información detallada de hoteles
- Políticas generales
- Servicios y amenidades

## Contribución

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## Acerca de

Asistente virtual especializado para VELAS RESORTS, diseñado para apoyar a los agentes de ventas con información precisa sobre hoteles, servicios y políticas. Integrado con Salesforce para datos en tiempo real y optimización del proceso de ventas.

## Recursos

- [Documentación de Ollama](https://ollama.ai/docs)
- [Documentación de Transformers](https://huggingface.co/docs/transformers/index)
- [Documentación de Mistral](https://mistral.ai/news/announcing-mistral-7b/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

## Estadísticas del Repositorio

![GitHub stars](https://img.shields.io/github/stars/hectorflores28/python-ollama-deepseek?style=social)
![GitHub forks](https://img.shields.io/github/forks/hectorflores28/python-ollama-deepseek?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/hectorflores28/python-ollama-deepseek?style=social)