import requests
from abc import ABC

class TelegramBot(ABC):

    def __init__(self, token):

        self.me = "https://api.telegram.org/bot{token}/sendMessage".format(token=token)
        self.payload = {'chat_id': "", 'text': "", 'parse_mode': 'HTML'}

    def sendMessage(self, message, chat):
        self.payload['chat_id'] = chat
        self.payload['text'] = message
        requests.post(self.me, data=self.payload)

class Terminator(TelegramBot):

    def __init__(self, token = "6642271737:AAE9DDJjRMbLD-l8FwrEs4H4EnclOMvM00c"):

        super().__init__(token)

    
    def sendMessage(self, message, chat = '109373025'):
        super().sendMessage(message + "\n\nHasta la vista!", chat) 