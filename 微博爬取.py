# weibo_single_script_modified_daily_shard.py
# 依赖: pip install requests lxml
# 在命令行运行: python weibo_single_script_modified_daily_shard.py

import os
import re
import csv
import time
import json
import random
import requests
from datetime import datetime, timedelta
from urllib.parse import quote, unquote
from lxml import html
from collections import OrderedDict, defaultdict

# ---- util functions (参考原项目逻辑，已简化) ----

def convert_weibo_type(weibo_type):
    if weibo_type == 0:
        return '&typeall=1'
    elif weibo_type == 1:
        return '&scope=ori'
    elif weibo_type == 2:
        return '&xsort=hot'
    elif weibo_type == 3:
        return '&atten=1'
    elif weibo_type == 4:
        return '&vip=1'
    elif weibo_type == 5:
        return '&category=4'
    elif weibo_type == 6:
        return '&viewpoint=1'
    return '&scope=ori'


def convert_contain_type(contain_type):
    if contain_type == 0:
        return '&suball=1'
    elif contain_type == 1:
        return '&haspic=1'
    elif contain_type == 2:
        return '&hasvideo=1'
    elif contain_type == 3:
        return '&hasmusic=1'
    elif contain_type == 4:
        return '&haslink=1'
    return '&suball=1'


def get_keyword_list(file_or_list):
    if isinstance(file_or_list, list):
        return file_or_list[:]
    if os.path.isfile(file_or_list):
        res = []
        with open(file_or_list, 'rb') as f:
            for line in f.read().splitlines():
                try:
                    s = line.decode('utf-8-sig').strip()
                except:
                    s = line.decode('utf-8', errors='ignore').strip()
                if s:
                    res.append(s)
        return res
    raise ValueError(f"既不是列表，也不是存在的文件: {file_or_list}")


def standardize_date(created_at):
    if not created_at:
        return ''
    s = str(created_at).strip()
    s = s.replace('\u200b', '').replace('\xa0', ' ').replace('\u3000', ' ').strip()
    now = datetime.now()
    try:
        if s == '刚刚':
            return now.strftime("%Y-%m-%d %H:%M")
        m = re.search(r'(\d+)\s*秒前', s)
        if m:
            sec = int(m.group(1))
            return (now - timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M")
        m = re.search(r'(\d+)\s*分钟前', s)
        if m:
            mnum = int(m.group(1))
            return (now - timedelta(minutes=mnum)).strftime("%Y-%m-%d %H:%M")
        if '半小时前' in s:
            return (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
        m = re.search(r'(\d+)\s*小时', s)
        if m:
            h = int(m.group(1))
            return (now - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
        if s.startswith('今天'):
            t = s.replace('今天', '').strip()
            if not t:
                t = '00:00'
            return f"{now.strftime('%Y-%m-%d')} {t}"
        if '昨天' in s:
            t = s.replace('昨天', '').strip()
            if not t:
                t = '00:00'
            return f"{(now - timedelta(days=1)).strftime('%Y-%m-%d')} {t}"
        if '前天' in s:
            t = s.replace('前天', '').strip()
            if not t:
                t = '00:00'
            return f"{(now - timedelta(days=2)).strftime('%Y-%m-%d')} {t}"

        m = re.search(r'(\d{4})\s*[年/-/]\s*(\d{1,2})\s*[月/-/]\s*(\d{1,2}).*?(\d{1,2}):(\d{1,2})', s)
        if m:
            y, mo, d, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}"

        m = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(\d{1,2}):(\d{1,2})', s)
        if m:
            mo, d, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            return f"{now.strftime('%Y')}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}"

        m = re.search(r'(\d{1,2})\s*月\s*(\d{1,2})\s*日', s)
        if m:
            mo, d = int(m.group(1)), int(m.group(2))
            return f"{now.strftime('%Y')}-{mo:02d}-{d:02d} 00:00"

        m = re.search(r'(\d{1,2})[-/](\d{1,2})\s+(\d{1,2}):(\d{1,2})', s)
        if m:
            mo, d, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            return f"{now.strftime('%Y')}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}"

        m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', s)
        if m and ':' not in s:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}-{mo:02d}-{d:02d} 00:00"

        try:
            dt = datetime.fromisoformat(s.replace(' ', 'T'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        return s
    except Exception:
        return created_at


def fetch_page(url, headers, timeout=15):
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        root = html.fromstring(r.text)
        return root
    except Exception:
        return None


def safe_text(el, xpath_expr):
    try:
        res = el.xpath(xpath_expr)
        if not res:
            return ''
        if isinstance(res, list):
            r = res[0]
        else:
            r = res
        if isinstance(r, str):
            return r.strip()
        return r if isinstance(r, str) else r.strip()
    except Exception:
        return ''


def extract_first_text(el, xpath_expr):
    v = el.xpath(xpath_expr)
    if not v:
        return ''
    if isinstance(v, list):
        txt = v[0]
    else:
        txt = v
    if hasattr(txt, 'text_content'):
        return txt.text_content().strip()
    return str(txt).strip()


def get_article_url_from_sel(sel):
    article_url = ''
    text = extract_first_text(sel, 'string(.)').replace('\u200b', '').replace('\ue627', '').replace('\n', '').replace(' ', '')
    if text.startswith('发布了头条文章'):
        urls = sel.xpath('.//a')
        for url in urls:
            icon = url.xpath('i[@class="wbicon"]/text()')
            if icon and icon[0] == 'O':
                href = url.xpath('@href')
                if href and href[0].startswith('http://t.cn'):
                    article_url = href[0]
                    break
    return article_url


def get_location_from_sel(sel):
    a_list = sel.xpath('.//a')
    for a in a_list:
        icon = a.xpath('./i[@class="wbicon"]/text()')
        if icon and icon[0] == '2':
            s = a.xpath('string(.)')
            if s:
                return s.strip()[1:]
    return ''


def get_at_users(sel):
    a_list = sel.xpath('.//a')
    at_list = []
    for a in a_list:
        hrefs = a.xpath('@href')
        texts = a.xpath('string(.)')
        if hrefs and texts:
            href = unquote(hrefs[0])
            text = texts.strip()
            if len(href) > 14 and len(text) > 1:
                if href[14:] == text[1:]:
                    at_user = text[1:]
                    if at_user not in at_list:
                        at_list.append(at_user)
    return ','.join(at_list)


def get_topics(sel):
    a_list = sel.xpath('.//a')
    topic_list = []
    for a in a_list:
        text = a.xpath('string(.)')
        if text and len(text) > 2 and text[0] == '#' and text[-1] == '#':
            t = text[1:-1]
            if t not in topic_list:
                topic_list.append(t)
    return ','.join(topic_list)


def get_vip(sel):
    vip_type = "非会员"
    vip_level = 0
    vip_container = sel.xpath('.//div[contains(@class,"user_vip_icon_container")]')
    if vip_container:
        imgs = vip_container[0].xpath('.//img/@src')
        if imgs:
            src = imgs[0]
            m = re.search(r'svvip_(\d+)\.png', src)
            if m:
                vip_type = "超级会员"
                vip_level = int(m.group(1))
            else:
                m2 = re.search(r'vip_(\d+)\.png', src)
                if m2:
                    vip_type = "会员"
                    vip_level = int(m2.group(1))
    return vip_type, vip_level


def get_ip_from_bid(bid, headers):
    if not bid:
        return ''
    url = f"https://weibo.com/ajax/statuses/show?id={bid}&locale=zh-CN"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return ''
        data = r.json()
        ip_str = data.get("region_name", "") or ''
        if ip_str:
            return ip_str.split()[-1]
        return ''
    except Exception:
        return ''


def parse_retweet_info(root_sel):
    try:
        ret_sel = root_sel.xpath('.//div[contains(@class,"card-comment")]')
        if not ret_sel:
            return '', '', ''
        block = ret_sel[0]
        block_mid = block.get('mid') or ''
        orig_mid = ''
        ad_nodes = block.xpath('.//*[@action-data]')
        for n in ad_nodes:
            ad = n.get('action-data') or ''
            m = re.search(r'mid=(\d+)', ad)
            if m:
                orig_mid = m.group(1)
                break
        orig_user_id = ''
        hrefs = block.xpath('.//a/@href')
        for href in hrefs:
            if not href:
                continue
            m1 = re.search(r'/u/(\d+)', href)
            if m1:
                orig_user_id = m1.group(1)
                break
            m2 = re.search(r'weibo\.com/(\d+)', href)
            if m2:
                orig_user_id = m2.group(1)
                break
        return block_mid or orig_mid, orig_user_id, orig_mid
    except Exception:
        return '', '', ''




def looks_like_time_text(s):
    if not s:
        return False
    s = str(s)
    patterns = [
        r'\d{4}[-/年]', r'\d{1,2}月\d{1,2}日', r'\d{1,2}[-/]\d{1,2}', r'\d{1,2}:\d{2}',
        r'刚刚', r'秒前', r'分钟前', r'小时', r'今天', r'昨天', r'半小时前'
    ]
    for p in patterns:
        if re.search(p, s):
            return True
    return False


def parse_created_and_source(sel):
    try:
        nodes = sel.xpath('.//*[contains(@class,"from")]') or sel.xpath('.//p[contains(@class,"from")]') or []
        for node in nodes:
            anchors = node.xpath('.//a')
            if anchors:
                for i, a in enumerate(anchors):
                    candidate = ''
                    for attr in ('title', 'date', 'data-time', 'data-datetime'):
                        if a.get(attr):
                            candidate = a.get(attr).strip()
                            break
                    if not candidate:
                        candidate = (a.text_content() or '').strip()
                    if candidate and looks_like_time_text(candidate):
                        src = ''
                        if i + 1 < len(anchors):
                            src = anchors[i + 1].text_content().strip()
                        else:
                            full = node.text_content()
                            if '来自' in full:
                                after = full.split('来自', 1)[1].strip()
                                src = after.split()[0] if after else ''
                        return candidate, src
            full_text = node.text_content().strip()
            if '来自' in full_text:
                parts = full_text.split('来自', 1)
                cand = parts[0].strip()
                src = parts[1].strip().split()[0] if parts[1].strip() else ''
                if cand and looks_like_time_text(cand):
                    return cand, src
            if looks_like_time_text(full_text):
                return full_text, ''
        full = sel.xpath('string(.)') or ''
        m = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}.*?\d{1,2}:\d{1,2})', full)
        if m:
            return m.group(1).strip(), ''
        m = re.search(r'(\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{1,2})', full)
        if m:
            return m.group(1).strip(), ''
        m = re.search(r'(\d{1,2}[-/]\d{1,2}\s*\d{1,2}:\d{1,2})', full)
        if m:
            return m.group(1).strip(), ''
        m = re.search(r'(刚刚|\d+秒前|\d+分钟前|半小时前|\d+小时前|今天|昨天|前天)', full)
        if m:
            return m.group(1).strip(), ''
        return '', ''
    except Exception:
        return '', ''


# ---- main scraper class ----

class WeiboScraper:
    def __init__(self, keywords, start_date, end_date, weibo_type=0, contain_type=0,
                 region_list=None, further_threshold=46, limit_result=0,
                 headers=None, output_dir='结果文件', flush_interval=100, max_open_files_per_keyword=20):
        self.keywords = get_keyword_list(keywords)
        # 保持关键词原样用于搜索，但用于文件名时做 decode+sanitize
        for i, k in enumerate(self.keywords):
            if len(k) > 2 and k[0] == '#' and k[-1] == '#':
                self.keywords[i] = '%23' + k[1:-1] + '%23'
        self.start_date = start_date
        self.end_date = end_date
        self.weibo_type = convert_weibo_type(weibo_type)
        self.contain_type = convert_contain_type(contain_type)
        self.region_list = region_list or ['全部']
        self.further_threshold = further_threshold
        self.limit_result = limit_result
        self.result_count = 0
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
        }
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # 写入缓冲/句柄相关
        self.flush_interval = int(flush_interval)
        self.max_open_files_per_keyword = int(max_open_files_per_keyword)
        # buffers: keyword -> date_str -> [rows]
        self.buffers = defaultdict(lambda: defaultdict(list))
        # file handles and csv writers: keyword -> OrderedDict(date_str -> (fileobj, writer))
        self.file_handles = defaultdict(OrderedDict)
        # header
        self.csv_header = [
            'id', 'bid', 'user_id', '用户昵称', '微博正文', '头条文章url',
            '发布位置', '艾特用户', '话题', '转发数', '评论数', '点赞数', '发布时间',
            '发布工具', '微博图片url', '微博视频url', 'retweet_id', 'retweet_user_id', 'retweet_mid', 'ip', 'user_authentication',
            '会员类型', '会员等级'
        ]

    def check_limit(self):
        if self.limit_result > 0 and self.result_count >= self.limit_result:
            print(f'已达到爬取结果数量限制：{self.limit_result} 条，停止爬取。')
            return True
        return False

    def sanitize_filename(self, s):
        if not s:
            return 'default'
        try:
            # 先尝试解码（若是 %23.. 之类）
            s = unquote(s)
        except Exception:
            pass
        # 替换路径/非法字符
        s = s.strip()
        s = re.sub(r'[\\/:*?"<>|]+', '_', s)
        # 剪短过长文件名
        if len(s) > 150:
            s = s[:150]
        return s

    def _open_csv_for_keyword_date(self, keyword, date_str):

        k = keyword
        d = date_str

        if d in self.file_handles[k]:
            fo, writer = self.file_handles[k][d]
            # mark as recently used
            self.file_handles[k].move_to_end(d)
            return fo, writer


        if len(self.file_handles[k]) >= self.max_open_files_per_keyword:
            old_date, (old_fo, old_writer) = self.file_handles[k].popitem(last=False)
            try:
                old_fo.close()
            except Exception:
                pass

        safe_kw = self.sanitize_filename(k)
        kw_dir = os.path.join(self.output_dir, safe_kw)
        os.makedirs(kw_dir, exist_ok=True)
        fname = f"{safe_kw}_{d}.csv"
        file_path = os.path.join(kw_dir, fname)
        first_write = not os.path.isfile(file_path)
        fo = open(file_path, 'a', encoding='utf-8-sig', newline='')
        writer = csv.writer(fo)
        if first_write:
            try:
                writer.writerow(self.csv_header)
                fo.flush()
            except Exception:
                pass
        self.file_handles[k][d] = (fo, writer)
        return fo, writer

    def _buffer_row_and_maybe_flush(self, keyword, date_str, row):
        self.buffers[keyword][date_str].append(row)
        if len(self.buffers[keyword][date_str]) >= self.flush_interval:
            self.flush_keyword_date(keyword, date_str)

    def flush_keyword_date(self, keyword, date_str):
        rows = self.buffers[keyword].get(date_str)
        if not rows:
            return
        fo, writer = self._open_csv_for_keyword_date(keyword, date_str)
        try:
            writer.writerows(rows)
            fo.flush()
        except Exception:
            pass
        self.buffers[keyword][date_str] = []

    def flush_all(self):
        for kw, date_map in list(self.buffers.items()):
            for d in list(date_map.keys()):
                if self.buffers[kw][d]:
                    try:
                        self.flush_keyword_date(kw, d)
                    except Exception:
                        pass
        for kw, od in list(self.file_handles.items()):
            for d, (fo, writer) in list(od.items()):
                try:
                    fo.close()
                except Exception:
                    pass
        self.file_handles = defaultdict(OrderedDict)
        self.buffers = defaultdict(lambda: defaultdict(list))

    def save_item_csv(self, item, keyword):
        ca = item.get('created_at', '') or ''
        if isinstance(ca, str) and ca:
            m = re.search(r'(\d{4}-\d{2}-\d{2})', ca)
            if m:
                date_part = m.group(1)
            else:
                try:
                    parsed = datetime.strptime(ca, '%Y-%m-%d %H:%M')
                    date_part = parsed.strftime('%Y-%m-%d')
                except Exception:
                    date_part = datetime.now().strftime('%Y-%m-%d')
        else:
            date_part = datetime.now().strftime('%Y-%m-%d')

        row = [
            item.get('id', ''),
            item.get('bid', ''),
            item.get('user_id', ''),
            item.get('screen_name', ''),
            item.get('text', ''),
            item.get('article_url', ''),
            item.get('location', ''),
            item.get('at_users', ''),
            item.get('topics', ''),
            item.get('reposts_count', ''),
            item.get('comments_count', ''),
            item.get('attitudes_count', ''),
            item.get('created_at', ''),
            item.get('source', ''),
            ','.join(item.get('pics', []) or []),
            item.get('video_url', ''),
            item.get('retweet_id', ''),
            item.get('retweet_user_id', ''),
            item.get('retweet_mid', ''),
            item.get('ip', ''),
            item.get('user_authentication', ''),
            item.get('vip_type', ''),
            item.get('vip_level', 0)
        ]
        self._buffer_row_and_maybe_flush(keyword, date_part, row)

    def parse_weibo_items_from_root(self, root, keyword):
        items = []
        nodes = root.xpath("//div[contains(@class,'card-wrap')]")
        for sel in nodes:
            if self.check_limit():
                break
            try:
                weibo = {}
                mid = sel.get('mid') or ''
                weibo['id'] = mid
                try:
                    hrefs = sel.xpath('.//div[@class="from"]/a[1]/@href')
                    bid = ''
                    if hrefs:
                        bid = hrefs[0].split('/')[-1].split('?')[0]
                except Exception:
                    bid = ''
                weibo['bid'] = bid
                try:
                    user_href = sel.xpath(".//div[@class='info']/div[2]/a/@href")
                    user_id = user_href[0].split('?')[0].split('/')[-1] if user_href else ''
                except:
                    user_id = ''
                weibo['user_id'] = user_id
                try:
                    screen_names = sel.xpath(".//div[@class='info']/div[2]/a/@nick-name")
                    screen_name = screen_names[0] if screen_names else extract_first_text(sel, ".//div[@class='info']/div[2]/a")
                except:
                    screen_name = ''
                weibo['screen_name'] = screen_name

                weibo['vip_type'], weibo['vip_level'] = get_vip(sel)

                txt_sel = sel.xpath('.//p[@class="txt"]')
                if txt_sel:
                    txt_sel = txt_sel[0]
                else:
                    txt_sel = sel

                content_full = sel.xpath('.//p[@node-type="feed_list_content_full"]')
                if content_full:
                    txt_sel = content_full[0]

                weibo_text = extract_first_text(txt_sel, 'string(.)').replace('\u200b', '').replace('\ue627', '')
                weibo['text'] = weibo_text

                weibo['article_url'] = get_article_url_from_sel(txt_sel)
                weibo['location'] = get_location_from_sel(txt_sel)
                weibo['at_users'] = get_at_users(txt_sel)
                weibo['topics'] = get_topics(txt_sel)

                def extract_count(xpath_expr):
                    try:
                        text = sel.xpath(xpath_expr)
                        if not text:
                            return '0'
                        t = text[0] if isinstance(text, list) else text
                        m = re.search(r'(\d+)', str(t))
                        return m.group(1) if m else '0'
                    except:
                        return '0'

                reposts = sel.xpath('.//ul[@class="act s-fr"]/li[1]//text()') or sel.xpath('.//a[contains(@action-type,"fl_list")]//text()')
                comments = sel.xpath('.//ul[@class="act s-fr"]/li[2]//text()') or sel.xpath('.//a[contains(@action-type,"cc_list")]//text()')
                attitudes = sel.xpath('.//ul[@class="act s-fr"]/li[3]//text()') or sel.xpath('.//a[contains(@action-type,"like_list")]//text()')

                weibo['reposts_count'] = extract_count(reposts)
                weibo['comments_count'] = extract_count(comments)
                weibo['attitudes_count'] = extract_count(attitudes)

                raw_created_at, source_text = parse_created_and_source(sel)
                weibo['created_at'] = standardize_date(raw_created_at)
                weibo['source'] = source_text or ''

                pics = []
                for img in sel.xpath('.//img[@src]'):
                    src = img.get('src') or ''
                    if src and src.startswith('http'):
                        if 'profile' in src or 'avatar' in src or 'logo' in src:
                            continue
                        pics.append(src)
                weibo['pics'] = list(dict.fromkeys(pics))
                video_url = ''
                vlinks = sel.xpath('.//a[@href and (contains(@href,"video") or contains(@href,"mp4") or contains(@href,"m3u8"))]/@href')
                if vlinks:
                    video_url = vlinks[0]
                else:
                    vsrc = sel.xpath('.//video/@src') or sel.xpath('.//div[@class="video_box"]//iframe/@src')
                    if vsrc:
                        video_url = vsrc[0]
                weibo['video_url'] = video_url

                retweet_id = ''
                retweet_user_id = ''
                retweet_mid = ''
                try:
                    block_mid_attr, orig_user_id, orig_mid = parse_retweet_info(sel)
                    retweet_id = block_mid_attr or ''
                    retweet_user_id = orig_user_id or ''
                    retweet_mid = orig_mid or ''
                except Exception:
                    retweet_id = ''
                    retweet_user_id = ''
                    retweet_mid = ''

                weibo['retweet_id'] = retweet_id
                weibo['retweet_user_id'] = retweet_user_id
                weibo['retweet_mid'] = retweet_mid

                user_auth = sel.xpath(".//div[contains(@class,'avator')]//svg/@id")
                ua = ''
                if user_auth:
                    ua = user_auth[0]
                if ua == 'woo_svg_vblue':
                    weibo['user_authentication'] = '蓝V'
                elif ua == 'woo_svg_vyellow':
                    weibo['user_authentication'] = '黄V'
                elif ua == 'woo_svg_vorange':
                    weibo['user_authentication'] = '红V'
                elif ua == 'woo_svg_vgold':
                    weibo['user_authentication'] = '金V'
                else:
                    weibo['user_authentication'] = '普通用户'

                weibo['ip'] = get_ip_from_bid(weibo.get('bid', ''), self.headers)

                items.append(weibo)
                self.result_count += 1
            except Exception:
                continue
        return items

    def run_for_keyword_timescope(self, keyword, base_url, start_str, end_str):
        url = f"{base_url}{self.weibo_type}{self.contain_type}&timescope=custom:{start_str}:{end_str}&page=1"
        while url:
            if self.check_limit():
                break
            root = fetch_page(url, self.headers)
            if root is None:
                time.sleep(random.uniform(1, 3))
                root = fetch_page(url, self.headers)
                if root is None:
                    print("请求失败或被拦截：", url)
                    break
            items = self.parse_weibo_items_from_root(root, keyword)
            for it in items:
                self.save_item_csv(it, keyword)
                if self.check_limit():
                    break
            next_links = root.xpath('//a[@class="next"]/@href')
            if next_links:
                next_href = next_links[0]
                if next_href.startswith('http'):
                    url = next_href
                else:
                    url = 'https://s.weibo.com' + next_href
            else:
                break
            time.sleep(random.uniform(1.0, 3.0))

    def run(self):
        for keyword in self.keywords:
            if self.check_limit():
                break
            print("开始关键词：", keyword)
            base_url = f"https://s.weibo.com/weibo?q={keyword}"
            first_url = f"{base_url}{self.weibo_type}{self.contain_type}&timescope=custom:{self.start_date}-0:{(datetime.strptime(self.end_date,'%Y-%m-%d')+timedelta(days=1)).strftime('%Y-%m-%d')}-0&page=1"
            root = fetch_page(first_url, self.headers)
            if root is None:
                print("打开首页失败，尝试直接按天拆分")
                need_split = True
            else:
                page_count_nodes = root.xpath('//ul[@class="s-scroll"]/li')
                page_count = len(page_count_nodes)
                need_split = page_count >= self.further_threshold

            if not need_split:
                start_str = self.start_date + '-0'
                end_dt = datetime.strptime(self.end_date, '%Y-%m-%d') + timedelta(days=1)
                end_str = end_dt.strftime('%Y-%m-%d') + '-0'
                self.run_for_keyword_timescope(keyword, base_url, start_str, end_str)
            else:
                sdate = datetime.strptime(self.start_date, '%Y-%m-%d')
                edate = datetime.strptime(self.end_date, '%Y-%m-%d')
                cur = sdate
                while cur <= edate:
                    if self.check_limit():
                        break
                    start_str = cur.strftime('%Y-%m-%d') + '-0'
                    cur_next = cur + timedelta(days=1)
                    end_str = cur_next.strftime('%Y-%m-%d') + '-0'
                    print(f"按天抓取: {keyword} {start_str} to {end_str}")
                    self.run_for_keyword_timescope(keyword, base_url, start_str, end_str)
                    cur = cur_next



if __name__ == '__main__':
    KEYWORD_LIST = ['武汉大学']
    START_DATE = '2025-08-17'
    END_DATE = '2025-08-18'
    WEIBO_TYPE = 0               # 0 全部, 1 原创, 2 热门, 3 关注人, 4 认证用户, 5 媒体, 6 观点
    CONTAIN_TYPE = 0             # 0 不筛选, 1 包含图片, 2 包含视频, 3 包含音乐, 4 包含短链接
    REGION = ['全部']
    FURTHER_THRESHOLD = 40     # 拆分门限
    LIMIT_RESULT = 0            # 爬取微博总数
    OUTPUT_DIR = 'E:/2025.9.10 微博与男女平权/result'
    FLUSH_INTERVAL = 100
    MAX_OPEN_FILES_PER_KEYWORD = 20

    COOKIE = 'SCF=AuW3M534DFJ3ART8fJc2akfVa3-nO2kp3bpmoGNDe2MloSMCpDA_emNd-_s-U_zCkqDTq7W5QgMnKYqK4j_NIjw.; SUB=_2A25FxUMKDeRhGeFG6lQW-SbLyz6IHXVmu9rCrDV8PUNbmtAYLRbDkW9NfiC685N-VPQR5HxkEg1ZTOMIVLHcQA1l; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WF6uRzD_2mq1q7KS7UEFh4S5NHD95QN1h2cS0.RS05EWs4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNS0npSoM41hM7entt; ALF=02_1760084058; _s_tentry=weibo.com; Apache=4217474210908.5674.1757492079159; SINAGLOBAL=4217474210908.5674.1757492079159; ULV=1757492079160:1:1:1:4217474210908.5674.1757492079159:; UOR=,,www.52runoob.com'  # 如果需要登录 cookie，可填入
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    if COOKIE:
        DEFAULT_HEADERS['cookie'] = COOKIE

    scraper = WeiboScraper(
        keywords=KEYWORD_LIST,
        start_date=START_DATE,
        end_date=END_DATE,
        weibo_type=WEIBO_TYPE,
        contain_type=CONTAIN_TYPE,
        region_list=REGION,
        further_threshold=FURTHER_THRESHOLD,
        limit_result=LIMIT_RESULT,
        headers=DEFAULT_HEADERS,
        output_dir=OUTPUT_DIR,
        flush_interval=FLUSH_INTERVAL,
        max_open_files_per_keyword=MAX_OPEN_FILES_PER_KEYWORD
    )

    try:
        scraper.run()
    except KeyboardInterrupt:
        print('\n检测到中断，正在保存缓冲并关闭文件...')
    except Exception as e:
        print('运行时出现异常：', e)
    finally:
        scraper.flush_all()
        print('爬取完成，结果保存在：', os.path.abspath(OUTPUT_DIR))
