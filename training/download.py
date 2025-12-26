from huggingface_hub import hf_hub_download

model_route = hf_hub_download(
    repo_id="ValenMarcial/MCP_Products",
    filename="unsloth.Q4_K_M.gguf",
    local_dir="./mi_modelo_ollama"
)