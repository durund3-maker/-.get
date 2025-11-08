
import os
import csv
import random
import time
import argparse
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


try:
    from webdriver_manager.chrome import ChromeDriverManager
    _WDM_AVAILABLE = True
except Exception:
    _WDM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


DEFAULT_CSV = "开题报告_图文_最新_一周内_1756206532.csv"
URL_COL = "笔记链接"
POSTED_LOG = "posted_urls.txt"
DEFAULT_COMMENTS = [
    "写得很好，受益匪浅！👍",
    "感谢分享，这部分讲得很清楚。",
    "很实用！谢谢你的笔记～",
    "内容质量不错，收藏了。😊",
    "好内容，点赞支持！"
]

MIN_DELAY = 8.0
MAX_DELAY = 25.0
LONG_BREAK_EVERY = 30
LONG_BREAK_MIN = 120
LONG_BREAK_MAX = 600
MAX_TRIES_PER_POST = 3


COMMENT_BOX_ID = "content-textarea"
SEND_BTN_CSS = "button.btn.submit"


FALLBACK_SEND_XPATHS = [
    "//button[contains(., '发送')]",
    "//button[contains(., '发布')]",
    "//button[contains(., '回复')]",
    "//span[contains(., '发送')]/ancestor::button",
    "//button[@type='submit']"
]


def random_comment(comments_list):
    base = random.choice(comments_list)
    if random.random() < 0.2:
        tails = [" 谢谢分享！", " 支持～", " :)", " 👍", " 哈哈"]
        base = base + random.choice(tails)
    return base

def load_csv_links(csv_path, url_col=URL_COL):
    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if url_col not in reader.fieldnames:
            raise ValueError(f"CSV 中没有名为 '{url_col}' 的列。可选列名：{reader.fieldnames}")
        for r in reader:
            url = (r.get(url_col) or "").strip()
            if url:
                rows.append({'url': url, 'row': r})
    return rows

def load_posted_log(path):
    s = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s.add(line.strip())
    return s

def append_posted_log(path, url):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(url + "\n")

def detect_verification_page(driver):
    txt = driver.page_source.lower()
    if "验证码" in txt or "verify" in txt or "请验证" in txt or "访问受限" in txt:
        return True
    return False


def create_driver(headless=False, user_data_dir=None, driver_path=None):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    if user_data_dir:
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

    service = None
    if driver_path:
        if not os.path.exists(driver_path):
            raise FileNotFoundError(f"指定的 chromedriver 路径不存在：{driver_path}")
        service = Service(driver_path)
        logging.info("使用手动指定的 chromedriver：%s", driver_path)
    else:
        if _WDM_AVAILABLE:
            try:
                path = ChromeDriverManager().install()
                service = Service(path)
                logging.info("webdriver_manager 下载并使用驱动：%s", path)
            except Exception as e:
                logging.warning("webdriver_manager 下载失败：%s", e)
                # service stays None, Selenium will try PATH
        if service is None:
            try:
                service = Service()  # Selenium 尝试从 PATH 找到 chromedriver
                logging.info("尝试使用系统 PATH 中的 chromedriver")
            except Exception as e:
                raise RuntimeError("无法获取 chromedriver：请安装 webdriver-manager 或者用 --driver-path 指定 chromedriver 可执行文件。") from e

    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

_JS_FILL = r"""
(function(text){
  function isVisible(e){ try{ return e && e.offsetParent !== null && e.clientHeight>0 && e.clientWidth>0; }catch(e){return false;} }
  var keywords = ['写评论','发表评论','说点什么','输入评论','回复'];
  var els = Array.from(document.querySelectorAll('[contenteditable], textarea, [role="textbox"], input'));
  // 优先 contenteditable 或包含关键词的
  var cand = els.filter(function(el){
    if(!isVisible(el)) return false;
    try{
      if(el.getAttribute && el.getAttribute('contenteditable')) return true;
      if(el.tagName && el.tagName.toLowerCase()==='textarea') return true;
      var txt = (el.getAttribute('placeholder')||'') + (el.innerText||'') + (el.value||'');
      for(var k of keywords){ if(txt.indexOf(k) !== -1) return true; }
    }catch(e){}
    return false;
  });
  if(cand.length === 0) cand = els.filter(isVisible);
  if(cand.length === 0) return {ok:false, reason:'not_found'};
  var el = cand[0];
  try{
    if(el.tagName && el.tagName.toLowerCase()==='textarea' || (el.tagName && el.tagName.toLowerCase()==='input')){
      el.focus(); el.value = text;
      el.dispatchEvent(new Event('input',{bubbles:true}));
    } else {
      el.focus(); el.innerText = text;
      el.dispatchEvent(new Event('input',{bubbles:true}));
    }
    return {ok:true};
  }catch(e){
    return {ok:false, reason:String(e)};
  }
})(arguments[0]);
"""

_JS_CLICK_SEND = r"""
(function(){
  var xpaths = [
    "//button[contains(., '发送')]",
    "//button[contains(., '发布')]",
    "//button[contains(., '回复')]",
    "//span[contains(., '发送')]/ancestor::button"
  ];
  function tryClick(el){
    try{ if(el && el.offsetParent !== null){ el.click(); return true; } }catch(e){}
    return false;
  }
  for(var xp of xpaths){
    try{
      var r = document.evaluate(xp, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
      for(var i=0;i<r.snapshotLength;i++){
        if(tryClick(r.snapshotItem(i))) return {ok:true};
      }
    }catch(e){}
  }
  // 尝试常规按钮文本匹配
  var buttons = Array.from(document.querySelectorAll('button,span,a'));
  for(var b of buttons){
    try{
      var txt = (b.innerText||'').trim();
      if(/发送|发布|回复/.test(txt) && b.offsetParent!==null){
        tryClick(b); return {ok:true};
      }
    }catch(e){}
  }
  return {ok:false};
})();
"""


def send_comment(driver, text, dry_run=True, wait_after=2.0):

    try:

        els = driver.find_elements(By.ID, COMMENT_BOX_ID)
        if els:
            el = els[0]

            driver.execute_script("arguments[0].innerHTML = arguments[1]; arguments[0].dispatchEvent(new Event('input',{bubbles:true}));", el, text)
            time.sleep(0.6 + random.random() * 0.7)

            try:
                send_btn = driver.find_element(By.CSS_SELECTOR, SEND_BTN_CSS)
                if dry_run:
                    logging.info("[dry-run] 找到发送按钮但不实际点击")
                    return True, "dry_run_button"
                else:
                    send_btn.click()
                    time.sleep(wait_after + random.random() * 1.0)
                    return True, "sent_by_button"
            except Exception:

                for xp in FALLBACK_SEND_XPATHS:
                    try:
                        btns = driver.find_elements(By.XPATH, xp)
                        if btns:
                            if dry_run:
                                logging.info("[dry-run] 找到发送按钮（fallback）但不实际点击")
                                return True, "dry_run_button_fallback"
                            btns[0].click()
                            time.sleep(wait_after + random.random() * 1.0)
                            return True, "sent_by_button_fallback"
                    except Exception:
                        continue

                try:
                    el.send_keys(Keys.ENTER)
                    time.sleep(wait_after + random.random() * 1.0)
                    return True, "sent_by_enter"
                except Exception:
                    # 最后回退到页面 JS 点击尝试
                    res = driver.execute_script(_JS_CLICK_SEND)
                    if isinstance(res, dict) and res.get('ok'):
                        return True, "sent_by_js_click"
                    return False, "no_send_button_found"
        else:

            res = driver.execute_script(_JS_FILL, text)
            if not (isinstance(res, dict) and res.get('ok')):
                return False, "js_fill_failed"

            res2 = driver.execute_script(_JS_CLICK_SEND)
            if isinstance(res2, dict) and res2.get('ok'):
                return True, "sent_by_js"
            return False, "js_click_failed"
    except Exception as e:
        logging.exception("send_comment 发生异常：%s", e)
        return False, "exception"


def post_comment_on_note(driver, note_url, comment_text, dry_run=True):
    logging.info("打开笔记：%s", note_url)
    try:
        driver.get(note_url)
    except Exception as e:
        logging.warning("打开页面异常：%s", e)
        return False, "open_failed"
    time.sleep(2 + random.random() * 1.8)

    if detect_verification_page(driver):
        logging.warning("检测到可能的验证/登录页面，暂停，请人工处理。")
        input("请在浏览器中完成验证或登录，完成后回车继续...")


    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.6);")
        time.sleep(1.0 + random.random() * 0.8)
    except Exception:
        pass


    try:
        # 按中文文本查找“评论”按钮
        comment_buttons = driver.find_elements(By.XPATH, "//*[contains(text(),'评论') and (self::button or self::span or self::a)]")
        for cb in comment_buttons:
            try:
                if cb.is_displayed():
                    cb.click()
                    time.sleep(0.8 + random.random() * 0.8)
                    break
            except Exception:
                continue
    except Exception:
        pass


    ok, reason = send_comment(driver, comment_text, dry_run=dry_run)
    if ok:
        logging.info("评论操作结果：%s", reason)
        return True, reason
    else:
        logging.warning("评论失败：%s，尝试回退策略", reason)

        try:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.6)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5 + random.random())
            ok2, reason2 = send_comment(driver, comment_text, dry_run=dry_run)
            if ok2:
                logging.info("回退后评论成功：%s", reason2)
                return True, reason2
            else:
                logging.warning("回退也失败：%s", reason2)
                return False, reason2
        except Exception as e:
            logging.exception("回退时异常：%s", e)
            return False, "fallback_exception"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--url-col", type=str, default=URL_COL)
    parser.add_argument("--comments-file", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-delay", type=float, default=MIN_DELAY)
    parser.add_argument("--max-delay", type=float, default=MAX_DELAY)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--user-data-dir", type=str, default=None)
    parser.add_argument("--driver-path", type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        logging.error("找不到 CSV 文件：%s", csv_path)
        return

    comments_list = DEFAULT_COMMENTS.copy()
    if args.comments_file and os.path.exists(args.comments_file):
        with open(args.comments_file, encoding='utf-8') as cf:
            lines = [l.strip() for l in cf if l.strip()]
            if lines:
                comments_list = lines

    posted = load_posted_log(POSTED_LOG)
    rows = load_csv_links(csv_path, url_col=args.url_col)
    logging.info("准备对 %d 个候选链接尝试评论（已排除 %d 个已评论）", len(rows), sum(1 for r in rows if r['url'] in posted))


    try:
        driver = create_driver(headless=args.headless, user_data_dir=args.user_data_dir, driver_path=args.driver_path)
    except Exception as e:
        logging.error("创建 WebDriver 失败：%s", e)
        if not _WDM_AVAILABLE and not args.driver_path:
            logging.error("提示：未检测到 webdriver-manager，且未指定 --driver-path，无法启动驱动。")
        return

    try:
        driver.get("https://www.xiaohongshu.com")
        logging.info("请在打开的浏览器中登录小红书（或使用 --user-data-dir 复用已登录配置）。")
        input("登录好后按回车继续（若已登录可直接回车）...")

        counter = 0
        for item in rows:
            url = item['url']
            if url in posted:
                logging.info("已记录为已评论，跳过：%s", url)
                continue
            counter += 1
            comment_text = random_comment(comments_list)

            success = False
            tries = 0
            while tries < MAX_TRIES_PER_POST and not success:
                tries += 1
                try:
                    ok, reason = post_comment_on_note(driver, url, comment_text, dry_run=args.dry_run)
                    if ok:
                        success = True
                        logging.info("处理完成（%s）: %s", reason, url)
                        posted.add(url)
                        append_posted_log(POSTED_LOG, url if not args.dry_run else (url + " [dryrun]"))
                        break
                    else:
                        logging.warning("尝试失败(%s)，重试 %d/%d", reason, tries, MAX_TRIES_PER_POST)
                        time.sleep(2 + random.random() * 2)
                except Exception as e:
                    logging.exception("处理该笔记时出错：%s", e)
                    time.sleep(2 + random.random() * 2)

            delay = random.uniform(max(0.1, args.min_delay), max(0.1, args.max_delay))
            logging.info("等待 %.1f 秒后继续（随机延时）", delay)
            time.sleep(delay)

            if counter % LONG_BREAK_EVERY == 0:
                lb = random.uniform(LONG_BREAK_MIN, LONG_BREAK_MAX)
                logging.info("已完成 %d 条，进行长休息 %.1f 秒", counter, lb)
                time.sleep(lb)

        logging.info("全部处理完毕")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
