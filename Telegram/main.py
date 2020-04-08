#!/usr/bin/python3.6
import telepot
import time
from slugify import slugify
import requests
bot = telepot.Bot('1003067694:AAFxSzzj4y_fl-5OcJPDeOSoEkzANknuqyY')
url = "http://127.0.0.1:8000/ask/"


def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)

    if content_type == 'text':

        slug = slugify(msg["text"])

        ans = requests.get(url + slug).json()

        #bot.sendMessage(chat_id, "slugify '{}'".format(slug))

        bot.sendMessage(chat_id, "Answer:\n"+" '{}'".format(ans['answer']))


bot.message_loop(handle)

print('Listening ...')

# Keep the program running.
while 1:
    time.sleep(10)
