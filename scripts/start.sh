#!/bin/bash

export $(grep -v '^#' .env | xargs)
REQUIRED_MODEL=$MODELO
VENV_DIR=".venv"
LOG_FILE="debug.log"

if [ -d "$VENV_DIR" ]; then
    source $VENV_DIR/bin/activate
else
    echo "❌ Error: No se encuentra la carpeta $VENV_DIR"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama no se encuentra en el sistema"
    echo "📥 Iniciando instalación automática"
    echo "🔑 Te pedirá tu contraseña para instalar"
    
    # Ejecutamos la instalación pero NO detenemos el script si da advertencias
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Verificación REAL: ¿Existe el comando después de instalar?
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama instalado correctamente"
    else
        echo "Error crítico: La instalación falló y no se encuentra el comando 'ollama'"
        exit 1
    fi
fi

# 3. FUNCIÓN PARA VERIFICAR ESTADO DEL SERVIDOR
check_ollama() {
    curl -s http://localhost:11434 > /dev/null
    return $?
}

OLLAMA_STARTED_BY_SCRIPT=false

# 4. GESTIÓN DEL SERVIDOR
if check_ollama; then
    echo "✅ El servidor Ollama ya está activo"
else
    echo "💤 El servidor no responde"
    echo "🔄 Iniciando una instancia temporal"
    
    ollama serve > "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    OLLAMA_STARTED_BY_SCRIPT=true

    echo "⏳ Arrancando el motor..."
    TIMEOUT=30
    COUNTER=0
    
    while ! check_ollama; do
        if [ $COUNTER -gt $TIMEOUT ]; then
            echo "Error: Tiempo de espera agotado"
            kill $OLLAMA_PID 2>/dev/null
            exit 1
        fi
        
        if ! kill -0 $OLLAMA_PID 2>/dev/null; then
            echo "Error: El proceso Ollama se cerró inesperadamente"
            cat "$LOG_FILE"
            exit 1
        fi
        
        sleep 1
        ((COUNTER++))
    done
    echo "✅ Instancia temporal lista"
fi

MODELFILE="Modelfile.avanzado" # O el nombre que uses

if [ -f "$MODELFILE" ]; then
    echo "📝 Forzando actualización del modelo '$REQUIRED_MODEL'..."
    ollama create "$REQUIRED_MODEL" -f "$MODELFILE"
else
    echo "⚠️  No encontré el Modelfile. Usando modelo en memoria."
fi

echo "--------------------------------------------------"
echo "🚀 Ejecutando main.py"
echo "--------------------------------------------------"
python main.py

if [ "$OLLAMA_STARTED_BY_SCRIPT" = true ]; then
    echo "--------------------------------------------------"
    echo "🛑 Apagando instancia temporal de Ollama..."
    kill $OLLAMA_PID
fi

echo "👋 ¡Hasta luego!"