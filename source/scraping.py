import requests

for i in range(15000, 20000):
    URL = "http://bts.barc.gov.in/EmpPhoto/" + str(i) + ".jpg"
   
    response = requests.get(URL)
    
    if response.status_code == 200:
        with open(str(i) + '_image.jpg', 'wb') as handler:
            handler.write(response.content)
        print(f"Image {i}.jpg downloaded successfully.")
    else:
        print(f"Image {i}.jpg not available (HTTP {response.status_code}).")