# Author: Fatih E. NAR
# Few-Shot Learning with GPT-J-6B using 16-Bits Quantization.
#
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import shutil
import traceback
import yaml
import re

# Define model name
model_name = 'EleutherAI/gpt-j-6B'

# Initialize tokenizer
tokenizer2 = AutoTokenizer.from_pretrained(model_name)
tokenizer2.pad_token = tokenizer2.eos_token

# Define local model directory
local_model_dir = './gpt-j-6b-cache'

# Clear the cache and re-download
shutil.rmtree(local_model_dir, ignore_errors=True)
os.makedirs(local_model_dir, exist_ok=True)

# Configure 16-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4"
)

# Explicitly download the model with 16-bit quantization
print("Downloading and loading model...")
model2 = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir=local_model_dir, 
    device_map="auto", 
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)
print("Model downloaded and loaded in 16-bit quantization.")

def extract_info(input_text):
    model2.eval()
    prompt = f"""Extract the network device and IP address from the following input:
Input: {input_text}
Output format:
Device: <device>
IP: <ip>

Output:
"""

    input_ids = tokenizer2.encode(prompt, return_tensors='pt', add_special_tokens=False).to(model2.device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model2.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1
        )
    
    extracted_text = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    extracted_info = extracted_text.split("Output:\n")[-1].strip()
    
    device_match = re.search(r'Device:\s*(\w+)', extracted_info)
    ip_match = re.search(r'IP:\s*(\S+)', extracted_info)
    
    device = device_match.group(1) if device_match else None
    ip = ip_match.group(1) if ip_match else None
    
    return device, ip

def generate_yaml2(input_text):
    device, ip = extract_info(input_text)
    
    if not device or not ip:
        return f"Error: Could not extract device or IP from input: {input_text}"
    
    # Split IP and prefix length
    ip_parts = ip.split('/')
    ip_address = ip_parts[0]
    prefix_length = ip_parts[1] if len(ip_parts) > 1 else '24'  # Default to /24 if not provided
    
    yaml_dict = {
        'interfaces': [
            {
                'name': device,
                'type': 'wlan' if device.startswith('wlan') else 'ethernet',
                'state': 'up',
                'ipv4': {
                    'enabled': True,
                    'dhcp': False,
                    'address': [
                        {
                            'ip': ip_address,
                            'prefix-length': int(prefix_length)
                        }
                    ]
                }
            }
        ]
    }
    
    return yaml.dump(yaml_dict, default_flow_style=False)

# Test examples
test_inputs2 = [
    "Configure the wlan12 interface with the static IPv4 address 38.6.220.100/20",
    "Configure the eth3 ethernet device with the static IPv4 address 232.162.200.174/25",
    "Set the eth1 ethernet device with the static IPv4 address 192.168.1.1/24",
    "Assign the eth2 ethernet device with the IPv4 address 10.0.0.1/8",
]

for test_input in test_inputs2:
    print(f"Input: {test_input}")
    try:
        print(f"Generated YAML:\n{generate_yaml2(test_input)}")
    except Exception as e:
        print(f"Error generating YAML: {str(e)}")
        print(traceback.format_exc())
    print()

print("Script execution completed.")
