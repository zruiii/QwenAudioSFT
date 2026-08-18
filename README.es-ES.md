# QwenAudioSFT
Códigos de reproducción para el ajuste fino de Qwen-Audio

Desde que el repositorio oficial [de Qwen-Audio](https://github.com/QwenLM/Qwen-Audio) no proporciona un script de ajuste fino, he recreado el código de ajuste fino para el modelo Qwen-Audio aquí, inspirándome en los scripts de ajuste fino de las anteriores series de Qwen. A continuación, se muestran algunos pasos para ayudarte a comenzar:

### Paso 1: Instalar dependencias
Primero, instale los paquetes necesarios ejecutando:
```bash
pip install -r requirements.txt
```

### Paso 2: Reemplazar el archivo original `modeling_qwen.py`
Estoy usando el modelo [Qwen-Audio-Chat](https://huggingface.co/Qwen/Qwen-Audio-Chat) para esto. Reemplace el archivo `modeling_qwen.py` del modelo original con la versión de este repositorio. Esto permite que el módulo AudioEncoder permanezca congelado durante el ajuste fino.

### Paso 3: Preparar tus datos
Por favor, formatea tus datos de entrenamiento similar al ejemplo siguiente:
```
sample = {
    "messages": [
        {
            "role": "user",
            "audio": "data/audio/T0055G0007S0001.wav",
            "content": "Please translate this audio into Chinese."
        },
        {
            "role": "assistant",
            "content": "没有人知道他为什么要这么做。"
        }
    ]
}
```

### Paso 4: Iniciar el entrenamiento
A continuación, actualiza las configuraciones en `finetune.sh` como lo harías al ajustar los modelos anteriores de la serie Qwen. Luego, inicia el script de entrenamiento con:
```bash
sh finetune.sh
```
