import requests
from bs4 import BeautifulSoup
import fake_useragent
import csv

user = fake_useragent.UserAgent().random
headers = {
    'user-agent': user
}


def get_html(url):
    r = requests.get(url, headers=headers)
    return r


def get_content(html):
    global price, name
    catalog = []
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('div', class_='catalog-section-item-wrapper')
    for i in items:
        price_items = i.find_all('div', class_='catalog-section-item-price')
        for j in price_items:
            price = j.find('span').get_text()
            price = price.replace(' ', '').replace('руб.', '').strip().replace('\xa0', '')
        name_items = i.find_all('div', class_='catalog-section-item-name')
        for j in name_items:
            name = j.find('a').get_text()

        catalog.append({
            'name': name,
            'price': price,
        })
    return catalog


def save_file(items, path):
    with open(path, 'w', encoding='utf8', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Названия товара', 'Цена товара'])
        for item in items:
            writer.writerow([item['name'], item['price']])


def parse():
    for URL in ['https://chebtf.ru/catalog/platya_zhenskie/', ]:
        html = get_html(URL)
        if html.status_code == 200:
            html = get_content(html.text)
        else:
            print('Error')
        filename = 'nsuparse.csv'
        save_file(html, filename)


parse()
