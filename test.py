import json
import requests

data = {
    "user": "alex",
    "path": "http://172.18.116.126/live?app=myapp&stream=test666"
}
r = requests.post('http://172.18.116.126/api/model/start', json=data)
print("start push stream*****************", r.status_code)

url = "http://172.18.116.126/api/model/start"

payload = json.dumps({
    "user": "alex",
    "path": "http://172.18.116.126/live?app=myapp&stream=test666"
})
headers = {
    'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)',
    'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.status_code)
