import json
import os

input_file = 'dataset.jsonl'
output_file = 'dataset_fixed.jsonl'

print(f"🔧 Reparando {input_file}...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    count = 0
    for line in infile:
        if not line.strip(): continue
        
        # 1. Leemos la línea completa (JSON externo)
        data = json.loads(line)
        
        # 2. Buscamos el mensaje del asistente
        for msg in data.get("messages", []):
            if msg.get("role") == "assistant":
                content_str = msg.get("content", "")
                
                # 3. Intentamos parsear el JSON interno (el que tiene "tool")
                try:
                    tool_data = json.loads(content_str)
                    
                    # 4. Hacemos el cambio de claves
                    changed = False
                    if "tool" in tool_data:
                        tool_data["name"] = tool_data.pop("tool")
                        changed = True
                    
                    if "tool_input" in tool_data:
                        tool_data["arguments"] = tool_data.pop("tool_input")
                        changed = True
                        
                    # 5. Guardamos el cambio si hubo alguno
                    if changed:
                        msg["content"] = json.dumps(tool_data, ensure_ascii=False)
                        count += 1
                        
                except json.JSONDecodeError:
                    # Si no es un JSON válido dentro del string, lo ignoramos
                    pass
        
        # 6. Escribimos la línea corregida en el nuevo archivo
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"✅ ¡Listo! Se corrigieron {count} líneas.")
print(f"📂 El archivo nuevo es: {output_file}")