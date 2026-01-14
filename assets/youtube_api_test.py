from googleapiclient.discovery import build
API_KEY = "AIzaSyBp3pLDcXzQXhAHVv_uOkoWP6Eu2ibkmSI"


youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

request = youtube.channels().list(
    part="snippet,statistics",
    forUsername="GoogleDevelopers"
)

response = request.execute()

print(response)
