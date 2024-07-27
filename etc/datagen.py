import random
import ipaddress
import json

def generate_ip(version=4):
    if version == 4:
        return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
    else:
        return str(ipaddress.IPv6Address(random.randint(0, 2**128 - 1)))

def generate_prefix_length(version=4):
    if version == 4:
        return random.randint(8, 32)
    else:
        return random.randint(16, 128)

def generate_interface_name():
    prefixes = ['eth', 'ens', 'enp', 'wlan', 'wlp', 'bond', 'br']
    return random.choice(prefixes) + str(random.randint(0, 99))

def generate_example(include_errors=False):
    interface = generate_interface_name()
    ip_version = random.choice([4, 6])
    ip = generate_ip(ip_version)
    prefix_length = generate_prefix_length(ip_version)
    
    if random.random() < 0.1 and include_errors:  # 10% chance of an error case
        error_type = random.choice(['invalid_ip', 'invalid_prefix', 'missing_ip', 'missing_prefix'])
        if error_type == 'invalid_ip':
            ip = '999.999.999.999' if ip_version == 4 else 'zzzz:zzzz:zzzz:zzzz:zzzz:zzzz:zzzz:zzzz'
        elif error_type == 'invalid_prefix':
            prefix_length = 999
        elif error_type == 'missing_ip':
            ip = ''
        elif error_type == 'missing_prefix':
            prefix_length = ''
    
    use_dhcp = random.random() < 0.2  # 20% chance of using DHCP
    
    input_str = f"Configure the {interface} interface with "
    if use_dhcp:
        input_str += "DHCP"
    else:
        input_str += f"the {'static ' if random.random() < 0.5 else ''}IPv{ip_version} address {ip}/{prefix_length}"
    
    output_str = f"---\ninterfaces:\n- name: {interface}\n  type: {interface[:3]}\n  state: up\n  ipv{ip_version}:\n    enabled: true\n"
    if use_dhcp:
        output_str += "    dhcp: true\n"
    else:
        output_str += f"    dhcp: false\n    address:\n    - ip: {ip}\n      prefix-length: {prefix_length}\n"
    
    return {"input": input_str, "output": output_str}

# Generate 1000 examples
expanded_data = [generate_example(include_errors=True) for _ in range(10000)]

# Write the data to a JSON file
with open('training_data.json', 'w') as f:
    json.dump(expanded_data, f, indent=2)

print("Data has been written to training_data.json")