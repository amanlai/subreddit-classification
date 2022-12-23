import requests
 
subject = '2022 FIFA World Cup Group H'
 
url = 'https://en.wikipedia.org/w/api.php'

params = {
            'action': 'parse',
            'page': subject,
            'format': 'json',
            'prop':'text',
            'redirects':''
        }
 
response = requests.get(url, params=params)
data = response.json()