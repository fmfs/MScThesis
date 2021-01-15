from six.moves import urllib

urllib.request.urlopen('http://127.0.0.1:8000/flights/flyFrom=LIS&returnTo=LIS&minDate=01/07/2019&maxDate=02/07/2019&cities=(BCN,CDG),(LHR)&duration=(1,1),(1)')