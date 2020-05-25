def MyAllies():

    import requests

    url = "https://myallies-breaking-news-v1.p.rapidapi.com/GetTopNews"

    headers = {
        'x-rapidapi-host': "myallies-breaking-news-v1.p.rapidapi.com",
        'x-rapidapi-key': "0df66faf3dmsh76de0bf6f69ec1bp110c29jsn42d2a5d01754"
        }

    response = requests.request("GET", url, headers=headers)

    print(response.text)
MyAllies()
