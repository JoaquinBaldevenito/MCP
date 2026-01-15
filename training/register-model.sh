#!/bin/bash

export $(grep -v '^#' .env | xargs)
MODEL=$MODELO

if [ -z "$MODEL" ]; then
    echo "❌ Error: La variable MODELO no está definida en el archivo .env"
    exit 1
fi
# 1. Encender Ollama en segundo plano
ollama serve > /dev/null 2>&1 &

# 2. Esperar 5 segundos a que arranque
echo "⏳ Esperando a que Ollama despierte..."
sleep 5

# 3. Registrar el modelo
if [ -f "Modelfile.avanzado" ]; then
    echo "📝 Buscando Modelfile.Avanzado en la subcarpeta..."
    ollama create $MODEL -f Modelfile.avanzado
else
    echo "📝 Registrando modelo desde el directorio actual..."
    ollama create $MODEL -f Modelfile
fi

# 4. Apagar Ollama para dejar todo limpio
pkill ollama

echo "✅ ¡Listo! Modelo '$MODEL' creado."