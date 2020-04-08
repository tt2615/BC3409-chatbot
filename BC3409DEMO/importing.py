import csv
from datetime import datetime
from main.models import Database

Database.objects.all().delete()
with open('question.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        question = list(row.values())[0]
        print(question)
        answer = list(row.values())[1]
        print(answer)
        interrogative = list(row.values())[2]
        print(interrogative)

        new_db = Database(question=question, answer=answer, interrogative=interrogative)
        new_db.save()

# exec(open('importing.py').read())
