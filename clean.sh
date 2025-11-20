#!/bin/bash

MODEL="mistral:instruct"

echo "🧹 INICIANDO LIMPIEZA..."
echo "--------------------------------"

if pgrep -x "ollama" > /dev/null; then
    echo "🗑️  Eliminando modelo '$MODEL'..."
    ollama rm "$MODEL"
    
    if [ $? -eq 0 ]; then
        echo "✅ Modelo eliminado y espacio liberado."
    else
        echo "⚠️  No se pudo eliminar el modelo (quizás ya no existe)."
    fi
else
    echo "⚠️  Ollama no está corriendo, no se puede desinstalar el modelo limpiamente."
    echo "   (Para borrarlo, primero debes iniciar Ollama)."
fi

echo "--------------------------------"

# 2. DETENER EL SERVICIO
if systemctl is-active --quiet ollama; then
    echo "🛑 Deteniendo servicio del sistema (te pedirá contraseña)..."
    sudo systemctl stop ollama
    echo "✅ Servicio detenido."
else
    echo "ℹ️  El servicio del sistema no estaba corriendo."
fi

# 3. MATAR PROCESOS SUELTOS
if pgrep -x "ollama" > /dev/null; then
    echo "🔪 Matando procesos residuales de Ollama..."
    pkill ollama
    echo "✅ Procesos terminados."
else
    echo "✅ No quedan procesos activos."
fi

echo "--------------------------------"
echo "👋 ¡Listo! Ollama está detenido y el modelo borrado."