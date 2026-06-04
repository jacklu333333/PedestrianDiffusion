from .common_imports import *


class TelegramCallback(Callback):
    def __init__(self, bot_token):
        self.bot_token = bot_token
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text):
        payload = {"chat_id": self.chat_id, "text": text}
        response = requests.post(self.api_url, data=payload)
        if response.status_code != 200:
            print("Failed to send Telegram message:", response.text)

    def on_train_start(self, trainer, pl_module):
        message = f"🚀 Training is starting on {socket.gethostname()}!"
        self.send_message(message)

    def on_train_end(self, trainer, pl_module):
        message = f"✅ Training is complete on {socket.gethostname()}!"
        self.send_message(message)

    def on_test_start(self, trainer, pl_module):
        message = f"🧪 Testing is starting on {socket.gethostname()}!"
        self.send_message(message)

    def on_test_end(self, trainer, pl_module):
        message = f"🧪 Testing is complete on {socket.gethostname()}!"
        self.send_message(message)
