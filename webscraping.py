import requests
import bs4
import lxml
import re
import json
"""
for i in range(2,20):

    hvg = 'https://hvg.hu/hvgmuerto/' + str(i) + '?ver=1'
    res = requests.get(hvg)
"""

def cleanhtml(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    cleantext = cleantext.replace('\n',' ') 
    return cleantext



def getLinksFromHVG(url_start, url_end, category_name, pagecount = 3):
    hvg = 'https://hvg.hu'

    pages = []
    for i in range(2, pagecount):
        url = url_start + str(i) + url_end
        res = requests.get(url)
        soup = bs4.BeautifulSoup(res.text, 'lxml')
        articles = soup.select('.column-articlelist > .articlelist-element > .text-holder > .heading-3 > a', href = True)

        for article in articles:
            link =  hvg + article.get('href')
            pages.append([category_name, getDataFromLink(link)])
        
    return pages

def getDataFromLink(url):
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.text, 'lxml')
    

    data = (soup.select('.article-main > .article-lead > p'))
    data += '\n'
    data += soup.select('.article-main > .article-content > p')
    
    
    
    mystring = ''

    for d in data:
        mystring += (str(d))
        mystring += ' '

    mystring = cleanhtml(mystring)
    return mystring
    
print("Started")
path = "C:\\Suli\\Tesztfeladat\\"
kultur_start = 'https://hvg.hu/hvgmuerto/'
kultur_end = '?ver=1'

sport_start = 'https://hvg.hu/sport/'
sport_end = '?ver=1'

techtud_start = 'https://hvg.hu/tudomany/'
techtud_end = '?ver=1'

kultur = getLinksFromHVG(kultur_start,kultur_end,'kultur',22)
sport = getLinksFromHVG(sport_start,sport_end,'sport',100)
techtud = getLinksFromHVG(techtud_start,techtud_end,'techtud',100)

data = []

for k in kultur:
    data.append(k)

for s in sport:
    data.append(s)

for t in techtud:
    data.append(t)


filename = path + 'data.json'

with open(filename, 'w', encoding='utf-8') as f:
    json.dump(data, f)


print("Finished")