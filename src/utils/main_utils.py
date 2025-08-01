import yaml


def read_yaml(file_path):
    with open(file_path,"r",encoding="utf-8") as file:
        config = yaml.safe_load(file)
        return config


def get_last_assistant_message(messages):
    for message in reversed(messages):
        if message['role'] == 'assistant':
            return message['content']
    return None

# Method 3: Get the last user message
def get_last_user_message(messages):
    for message in reversed(messages):
        if message['role'] == 'user':
            return message['content']
    return None


def get_all_user_message(message):
    content = []
    for msg in message:
        if msg['role'] == "user":
            content.append(msg['content'])
    return content


def get_all_ai_message(message):
    content = []
    for msg in message:
        if msg['role'] == "assistant":
            content.append(msg['content'])
    return content


def get_all_corect_message(message):
    content = []
    for msg in message:
        if msg['role'] == "correct_answer":
            content.append(msg['content'])
    return content

