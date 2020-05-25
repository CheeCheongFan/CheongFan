from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='b3648319d9264d0a918ea4e75a6c968f')
pages = 1

def getall(pages):
    results = newsapi.get_everything(q='economy',
                                from_param='2020-04-11',
                                to='2020-05-08',
                                language='en',
                                sort_by='relevancy',
                                page=pages)
    return results

x = [1:21]
getall



