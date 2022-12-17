import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# url = 'https://ungaoyk0ki.execute-api.us-west-2.amazonaws.com/test/predict'

data = {'url': 'https://bit.ly/3PzCqJ2'}

result = requests.post(url, json=data).json()
print(result)
