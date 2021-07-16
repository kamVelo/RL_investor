import requests as rq
from time import sleep
symbol = 'F'
urls = {
    'STOCH': 'https://www.alphavantage.co/query?function=STOCH&symbol='+symbol+'&interval=60min&apikey=dsfdo&outputsize=full&datatype=csv',
    '20EMA': 'https://www.alphavantage.co/query?function=EMA&symbol='+symbol+'&interval=60min&time_period=20&series_type=close&apikey=sdfsf&outputsize=full&datatype=csv',
    '50EMA': 'https://www.alphavantage.co/query?function=EMA&symbol='+symbol+'&interval=60min&time_period=50&series_type=close&apikey=sdfsf&outputsize=full&datatype=csv',
    '200EMA': 'https://www.alphavantage.co/query?function=EMA&symbol='+symbol+'&interval=60min&time_period=200&series_type=close&apikey=sdfsf&outputsize=full&datatype=csv',
    'RSI': 'https://www.alphavantage.co/query?function=RSI&symbol='+symbol+'&interval=60min&time_period=14&series_type=close&apikey=sadfdf&datatype=csv&outputsize=full',
    'hourly': 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+symbol+'&interval=60min&apikey=sjdfd&datatype=csv&outputsize=full',
    'bbands': 'https://www.alphavantage.co/query?function=BBANDS&symbol='+symbol+'&interval=60min&time_period=20&series_type=close&nbdevup=2&nbdevdn=2&apikey=dffg&outputsize=full&datatype=csv',
    'vwap': 'https://www.alphavantage.co/query?function=VWAP&symbol='+symbol+'&interval=60min&apikey=asdflfj&datatype=csv&outputsize=full'
}
counter = 0
for item in urls:
    if counter % 5 == 0 and counter != 0:
        sleep(60)
    filename = symbol + '_' + item + '.csv'
    data = rq.get(urls[item]).content
    file = open(filename, "wb")
    file.write(data)
    file.close()
    counter += 1
