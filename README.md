# EJECUCION

## Ejecutar el MCP
```sh
chmod +x start.sh
./start.sh
```


## Para eliminar el modelo
```sh
chmod +x clean.sh
./clean.sh
```

## Para registrar un modelo
```sh
chmod +x register-model.sh
./register-model.sh
```

## Dependencias necesarias

```sh
sudo apt-get update && sudo apt-get install build-essential cmake
sudo apt-get update && sudo apt-get install libcurl4-openssl-dev

```

## Flujo de Ejecucion para entrenamiento
### Crea la carpeta mi _modelo_afinado_lora
```sh
python train.py
```

### Crea el .gguf para ollama
```sh
python export.py
```

### Registra el modelo entrenado anteriormente
```sh
./register-model.sh
```

### Ejecutar main
```sh
./start.sh
```



