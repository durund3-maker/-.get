import requests
import csv
import time
import random
import datetime

temp = datetime.datetime.now()
time2 = temp.strftime("%Y-%m-%d")
f = open(f'E:/实时股票数据/{time2}.csv', mode='w', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(f, fieldnames=[
    '股票代码', '股票名称', '当前价', '涨跌额', '涨跌幅', '年初至今',
    '成交量', '成交额', '换手率', '市盈率', '股息率', '市值',
])
csv_writer.writeheader()

ua_list = [

]

cookie = ''

for page in range(1, 167):
    print(f'正在爬取第 {page} 页数据...')

    headers = {
        'cookie': cookie,
        'user-agent': random.choice(ua_list),
        'referer': 'https://xueqiu.com/hq',
    }

    url = f'https://stock.xueqiu.com/v5/stock/screener/quote/list.json?page={page}&size=30&order=desc&order_by=percent&market=CN&type=sh_sz'

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f'第 {page} 页请求失败，状态码 {response.status_code}')
            continue

        json_data = response.json()
        info_list = json_data.get('data', {}).get('list', [])

        for index in info_list:
            dit = {
                '股票代码': index['symbol'],
                '股票名称': index['name'],
                '当前价': index['current'],
                '涨跌额': index['chg'],
                '涨跌幅': index['percent'],
                '年初至今': index['current_year_percent'],
                '成交量': index['volume'],
                '成交额': index['amount'],
                '换手率': index['turnover_rate'],
                '市盈率': index['pe_ttm'],
                '股息率': index['dividend_yield'],
                '市值': index['market_capital'],
            }
            csv_writer.writerow(dit)

        time.sleep(random.uniform(1.5, 4))

    except Exception as e:
        print(f'第 {page} 页出错：{e}')
        time.sleep(random.uniform(3, 6))

f.close()
print("数据采集完成")
