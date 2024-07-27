# Author: Fatih E. NAR
# No_AI but simple text parser and generator
#
import yaml

def extract_info(input_text):
    
    # This part would use the fine-tuned model to extract entities
    # For demonstration, we'll use a simplified extraction
    interface = input_text.split()[2]
    ip = input_text.split()[-1].split('/')[0]
    prefix = input_text.split('/')[-1]
    
    return interface, ip, prefix

def generate_yaml_config(interface, ip, prefix):
    yaml_dict = {
        'interfaces': [
            {
                'name': interface,
                'type': 'wlan' if interface.startswith('wlan') else 'ethernet',
                'state': 'up',
                'ipv4': {
                    'enabled': True,
                    'dhcp': False,
                    'address': [
                        {
                            'ip': ip,
                            'prefix-length': int(prefix)
                        }
                    ]
                }
            }
        ]
    }
    return yaml.dump(yaml_dict, default_flow_style=False)

def generate_network_config(input_text):
    interface, ip, prefix = extract_info(input_text)
    return generate_yaml_config(interface, ip, prefix)

# Test examples
test_inputs = [
    "Configure the wlan12 interface with the static IPv4 address 38.6.220.100/20",
    "Configure the eth3 ethernet device with the static IPv4 address 232.162.200.174/25",
    "Set the eth1 ethernet device with the static IPv4 address 192.168.1.1/24",
    "Assign the eth2 ethernet device with the IPv4 address 10.0.0.1/8",
]

for test_input in test_inputs:
    print(f"Input: {test_input}")
    print(f"Generated YAML:\n{generate_network_config(test_input)}")
    print()

print("Script execution completed.")
