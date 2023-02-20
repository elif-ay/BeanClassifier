import requests
import os

fileName = "C:\\Users\\Elif\\Desktop\\bitirmeProjesi\\beyazTohumlarİlk\\"
outputPath = "C:\\Users\\Elif\\Desktop\\bitirmeProjesi\\beyazTohumlarSon\\"

for r, d, f in os.walk(fileName):
    for file in f:
        if file.endswith(".JPG"):
            imageName = os.path.join(r, file)
            response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(imageName, 'rb')},
            data={'size': 'auto'},
            headers={'X-Api-Key': 'e1LAKhJJte7uE4BPrqYpCude'},)
            if response.status_code == requests.codes.ok:
              output = outputPath + file
              with open(output, 'wb') as out:
                out.write(response.content)
            else:
              print("Error:", response.status_code, response.text)
