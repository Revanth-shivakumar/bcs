import requests 

resp = requests.post("http://localhost:2000/predict",files={'file': open('test/1.jpg', 'rb')})
print(resp.text)