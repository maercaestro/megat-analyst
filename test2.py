import requests

url = 'https://webservice.gvsi.com/api/v3/getdaily?symbols=%40Naphtha_Crack&fields=close%2Ctradedatetimeutc&output=json&includeheaders=true&startdate=01%2F18%2F2025&enddate=02%2F01%2F2025'

payload=""""""
headers = {
'Authorization': 'Basic UEVUQVBJMzpadm1GWGZmWA=='
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
