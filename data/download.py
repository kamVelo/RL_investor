import requests as rq
from time import sleep
symbol = 'TSM'
urls = {
    'STOCH' : 'https://www.alphavantage.co/query?function=STOCH&symbol=F&interval=60min&apikey=dsfdo&outputsize=full&datatype=csv',
    '20EMA' : 'https://www.alphavantage.co/query?function=EMA&symbol=F&interval=60min&time_period=20&series_type=close&apikey=sdfsf&outputsize=full&datatype=csv',
    '50EMA' : 'https://www.alphavantage.co/query?function=EMA&symbol=F&interval=60min&time_period=50&series_type=close&apikey=sdfsf&outputsize=full&datatype=csv',
    '200EMA' : 'https://www.alphavantage.co/query?function=EMA&symbol=F&interval=60min&time_period=200&series_type=close&apikey=sdfsf&outputsize=full&datatype=csv',
    'RSI' : 'https://www.alphavantage.co/query?function=RSI&symbol=F&interval=60min&time_period=14&series_type=close&apikey=sadfdf&datatype=csv&outputsize=full',
    'hourly' : 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=F&interval=60min&apikey=sjdfd&datatype=csv&outputsize=full'
}
counter = 0
for item in urls:
    if counter % 5 == 0 and counter != 0:
        sleep(30)
    filename = symbol + '_' + item + '.csv'
    data = rq.get(urls[item]).content
    file = open(filename, "wb")
    file.write(data)
    file.close()
    counter += 1
