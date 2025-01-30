## DeepSeek Models on Openshift

(1) Use ollama and openwebui deployment manifests to deploy both ollama and openwebui in same namespace. <br>
(2) Go to ollama pod terminal and pull the deepseek model that will fit on your system.<br>

<div align="center">
    <img src="https://raw.githubusercontent.com/tme-osx/Telco-AIX/refs/heads/main/etc/deepseek/images/ollama.png" width="800"/>
</div>

## Sandbox
Our Sandbox is a Single Node Openshift (12 Cores Xeon with 128GB Memory) with Dual NVIDIA RTX 4090 
<div align="center">
    <img src="https://raw.githubusercontent.com/tme-osx/Telco-AIX/refs/heads/main/etc/deepseek/images/smi.png" width="1200"/>
</div>

## Test
Ask AI for the meaning of life :-)
<div align="center">
    <img src="https://raw.githubusercontent.com/tme-osx/Telco-AIX/refs/heads/main/etc/deepseek/images/42.png" width="1200"/>
</div>

### References:
(A) https://ollama.com/library/deepseek-r1 (Model Size Range 1.1G <-> 404GB)  <br>
(B) https://ollama.com/library/deepseek-v3 (Model Size 404GB)
