#!/usr/bin/env python
import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://raw.githubusercontent.com/Optimistix/Traffic_Sign_Recognition_using_Deep-Learning/main/00051.png'}

result = requests.post(url, json=data).json()
print(result)
