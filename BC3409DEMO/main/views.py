import csv
import numpy as np
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render, render_to_response
from django.shortcuts import render
import main

from django.http import HttpResponse
import simplejson as json

# Create your views here.
# import it
from django.http import JsonResponse

from .models import Database

import pandas as pd

# Vector comparison module
import os
from . import query
from django.shortcuts import render
from django.http import HttpResponse
import simplejson as json
from django.http import HttpResponse
import csv
# Create your views here.
# import it
from django.core.mail import send_mail

from django.http import JsonResponse
size = 0

def askchatbot(request, slug):
    # global question_to_vector
    # global question_to_answer
    # global vector_to_question
    global additionalRes

    # do something with the your data
    print('System is processing your question...')
    unslug = slug.replace("-", " ")
    print('query content: ' + unslug)
    result = Database.objects.all().values()
    df = pd.DataFrame(result)

    if unslug == ('y' or 'Y'):
        print('enter yes')
        response_data = {'answer': 'Ask me another question if you want'}

    elif unslug == ('n' or 'N') and len(additionalRes)>0:
        print('enter no')
        addQn = ''
        i = 1
        for res in additionalRes:
            if i > 10:
                break
            addQn += str(i) + ': ' + res.question + '\n'
            i += 1
        print(addQn)

        response_data = {'answer': 'Do you mean any of the questions below: \n' + addQn + '\nPlease enter question number to see the answer'}

    elif unslug.isdigit() and len(additionalRes)>0:
        print('enter qns selection')
        index = int(unslug)
        print(index)
        selection = additionalRes[index-1]
        print(selection)
        response_data = {'answer': selection.answer + '\n\nKey in another question number to see more answer or ask me a new question'}

    else:
        if size == df.shape[0]:
            pass
        else:
            query.load_csv_into_memory(df)
        #     response = HttpResponse (content_type='text/csv')
        #     writer = csv.writer(open("qna.csv", "w", encoding='utf-8',newline=''))
        #     for item in Database.objects.values_list('question','answer'):
        #         writer.writerow(item)
        #     question_to_vector, question_to_answer, vector_to_question = load_csv_into_memory("C:\\Users\\KangYu\\Desktop\\BC3409\\BC3409DEMO\\qna.csv","utf-8")
        input_qn = query.question(unslug)
        result = query.query(input_qn, True)
        additionalRes = result['additionalRes']
        print(len(additionalRes))

        if result["sim"] < 0.5:
            response_data = {'answer': "Answer may not be what you want but we are working on it!\n" + result["answer"]}

            send_mail(
                'UNKNOWN QN REPLY ASAP',
                unslug,
                'bc3409-4fddf9@inbox.mailtrap.io',
                ['bc3409-4fddf9@inbox.mailtrap.io'],
                fail_silently=False,
                )

        response_data = {'answer': result["answer"] + '\n\nIs this what you want?\nkey in Y for Yes to answer another question\nkey in N for No to view similar questions.'}

    return HttpResponse(json.dumps(response_data), content_type="application/json")


@csrf_exempt
def get_response(request):
    response = {'status': None}

    if request.method == 'POST':
        data = json.loads(request.body)
        message = data['message']

        result = Database.objects.all().values()
        df = pd.DataFrame(result)

        if size == df.shape[0]:
            pass
        else:
            query.load_csv_into_memory(df)

        input_qn = query.question(message)
        result = query.query(input_qn, True)

        if result["sim"] < 0.5:
            chat_response = {'answer': "Answer may not be what you want but we are working on it!\n" + result["answer"]}
            send_mail(
                'UNKNOWN QN REPLY ASAP',
                message,
                'bc3409-4fddf9@inbox.mailtrap.io',
                ['bc3409-4fddf9@inbox.mailtrap.io'],
                fail_silently=False,
                )

        else:
            chat_response = {'answer': result["answer"]}


        response['message'] = {'text': chat_response["answer"], 'user': False, 'chat_bot': True}
        response['status'] = 'ok'

    else:
        response['error'] = 'no post data found'

    return HttpResponse(
        json.dumps(response),
        content_type="application/json"
    )


def home(request, template_name="home.html"):
    context = {'title': 'Chatbot Version 1.0'}
    return render_to_response(template_name, context)
