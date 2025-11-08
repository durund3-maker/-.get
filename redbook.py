
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
from selenium.common.exceptions import NoSuchElementException, WebDriverException
try:
    from webdriver_manager.chrome import ChromeDriverManager
    _WDM_AVAILABLE = True
except Exception:
    _WDM_AVAILABLE = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_CSV = "E:/开题报告_图文_最新_一天内.csv"
URL_COL = "笔记链接"
POSTED_LOG = "posted_urls.txt"
DEFAULT_COMMENTS = [
    "不怕不怕[大笑R][大笑R]，开题报告速成大法，约个时间，一个小时教会你",
    "加油[笑哭R]开题按开题架构写就行了，可能难一点的就是文献综述和线路图",
    "不怕不怕[大笑R][大笑R]，开题报告速成大法，约个时间，一个小时教会你",
    "开题报告没有那么难，多看几篇论文，然后参考构思一下自己的论文框架。通过觉得太麻烦就踹我一下[吧唧R][吧唧R]"
 ]
MIN_DELAY = 8.0
MAX_DELAY = 20.0
LONG_BREAK_EVERY = 30
LONG_BREAK_MIN = 120
LONG_BREAK_MAX = 600
MAX_TRIES_PER_POST = 3
COMMENT_BOX_ID = "content-textarea"
SEND_BTN_CSS = "button.btn.submit"

def random_comment(comments_list):
    base = random.choice(comments_list)
    if random.random() < 0.25:
        tails = [" 谢谢分享！", " 支持～", " :)", " 👍"]
        base += random.choice(tails)
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
    if service is None:
        service = Service()
        logging.info("尝试使用 PATH 中的 chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def ensure_note_viewable(driver, max_retries=4, hide_overlay_if_fail=True):
    overlay_phrases = [
        "请用APP打开", "请用 App 打开", "请用小红书APP打开", "暂时无法浏览",
        "暂时无法访问", "该内容暂时无法查看", "请在app中查看", "请前往App查看",
        "访问受限", "请在App中查看"
    ]
    try:
        try:
            els = driver.find_elements(By.ID, COMMENT_BOX_ID)
            for e in els:
                try:
                    if e.is_displayed():
                        return True
                except Exception:
                    continue
        except Exception:
            pass

        page = driver.page_source or ""
        found = any(ph in page for ph in overlay_phrases)
        if not found:
            return True

        for attempt in range(1, max_retries + 1):
            logging.info("检测到覆盖提示，刷新以尝试恢复 (%d/%d)...", attempt, max_retries)
            try:
                driver.refresh()
            except Exception as e:
                logging.debug("refresh 出错：%s", e)
            sleep_t = (1.3 ** attempt) + random.uniform(0.5, 1.2)
            time.sleep(sleep_t)
            try:
                els = driver.find_elements(By.ID, COMMENT_BOX_ID)
                for e in els:
                    try:
                        if e.is_displayed():
                            logging.info("刷新后发现评论输入框，页面恢复")
                            return True
                    except Exception:
                        continue
            except Exception:
                pass
            page = driver.page_source or ""
            if not any(ph in page for ph in overlay_phrases):
                logging.info("刷新后未发现覆盖提示，页面恢复")
                return True

        if hide_overlay_if_fail:
            try:
                logging.info("尝试隐藏覆盖层（最后手段）")
                js = """
                (function(phrases){
                  function isVisible(e){ try{ return e && e.offsetParent !== null && e.clientHeight>0 && e.clientWidth>0; }catch(e){return false;} }
                  var found=false;
                  phrases.forEach(function(p){
                    try {
                      var nodes = document.evaluate("//*[contains(normalize-space(.), '"+p+"')]", document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                      for(var i=0;i<nodes.snapshotLength;i++){
                        var el = nodes.snapshotItem(i);
                        if(isVisible(el)){ el.style.display='none'; found=true; }
                      }
                    } catch(e){}
                  });
                  return found;
                })(arguments[0]);
                """
                hidden = driver.execute_script(js, overlay_phrases)
                if hidden:
                    time.sleep(1.0 + random.random() * 0.8)
                    try:
                        els = driver.find_elements(By.ID, COMMENT_BOX_ID)
                        for e in els:
                            try:
                                if e.is_displayed():
                                    logging.info("隐藏覆盖层后发现评论输入框，页面恢复")
                                    return True
                            except Exception:
                                continue
                    except Exception:
                        pass
            except Exception as e:
                logging.debug("隐藏 overlay 出错：%s", e)

        logging.warning("多次尝试后页面仍不可用")
        return False
    except Exception as e:
        logging.exception("ensure_note_viewable 异常：%s", e)
        return False

def get_wrapper_element(driver, selector):
    try:
        el = driver.find_element(By.CSS_SELECTOR, selector)
        return el
    except NoSuchElementException:
        return None

def read_wrapper_use_href_and_count(wrapper_el):
    try:
        href = None
        count = None
        # find svg/use within wrapper
        try:
            # find <use> descendant
            use_el = wrapper_el.find_element(By.XPATH, ".//*[local-name()='use' or name()='use']")
            href = use_el.get_attribute("xlink:href") or use_el.get_attribute("href") or use_el.get_attribute("hrefx")
        except Exception:
            href = None
        # find span.count within wrapper or following sibling
        try:
            span = wrapper_el.find_element(By.CSS_SELECTOR, "span.count")
            txt = span.text.strip()
            if txt.isdigit():
                count = int(txt)
        except Exception:
            # try following-sibling span
            try:
                fs = wrapper_el.find_elements(By.XPATH, "following-sibling::span[contains(@class,'count')]")
                if fs:
                    t = fs[0].text.strip()
                    if t.isdigit():
                        count = int(t)
            except Exception:
                count = None
        return href, count
    except Exception:
        return None, None

def click_element_visible(driver, el, dry_run=False):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.3 + random.random() * 0.5)
        if dry_run:
            return True
        try:
            el.click()
            return True
        except Exception:
            try:
                driver.execute_script("arguments[0].click();", el)
                return True
            except Exception:
                return False
    except Exception:
        return False

def like_wrapper_action(driver, dry_run=True, max_retries=3):
    try:
        wrapper = get_wrapper_element(driver, ".like-wrapper")
        if not wrapper:
            return False, "no_like_wrapper"
        href_before, count_before = read_wrapper_use_href_and_count(wrapper)
        if href_before and "#liked" in href_before:
            return True, f"already_liked ({count_before})"
        if dry_run:
            return True, f"dry_run_would_like ({href_before},{count_before})"
        ok_click = click_element_visible(driver, wrapper, dry_run=False)
        if not ok_click:
            return False, "click_failed"
        for _ in range(max_retries):
            time.sleep(0.6 + random.random() * 0.8)
            wrapper2 = get_wrapper_element(driver, ".like-wrapper")
            if not wrapper2:
                continue
            href_after, count_after = read_wrapper_use_href_and_count(wrapper2)
            if href_after and "#liked" in href_after:
                return True, f"liked_confirmed {count_before}->{count_after}"
            if count_before is not None and count_after is not None and count_after > count_before:
                return True, f"liked_by_count {count_before}->{count_after}"
        return False, "like_not_confirmed"
    except Exception as e:
        logging.exception("like_wrapper_action 异常：%s", e)
        return False, "exception"

def collect_wrapper_action(driver, dry_run=True, max_retries=3):
    try:
        wrapper = get_wrapper_element(driver, ".collect-wrapper")
        if not wrapper:
            return False, "no_collect_wrapper"
        href_before, count_before = read_wrapper_use_href_and_count(wrapper)
        if href_before and "#collected" in href_before:
            return True, f"already_collected ({count_before})"
        if dry_run:
            return True, f"dry_run_would_collect ({href_before},{count_before})"
        ok_click = click_element_visible(driver, wrapper, dry_run=False)
        if not ok_click:
            return False, "click_failed"
        for _ in range(max_retries):
            time.sleep(0.6 + random.random() * 0.8)
            wrapper2 = get_wrapper_element(driver, ".collect-wrapper")
            if not wrapper2:
                continue
            href_after, count_after = read_wrapper_use_href_and_count(wrapper2)
            if href_after and "#collected" in href_after:
                return True, f"collected_confirmed {count_before}->{count_after}"
            if count_before is not None and count_after is not None and count_after > count_before:
                return True, f"collected_by_count {count_before}->{count_after}"
        return False, "collect_not_confirmed"
    except Exception as e:
        logging.exception("collect_wrapper_action 异常：%s", e)
        return False, "exception"

_JS_FILL = r"""
(function(text){
  function isVisible(e){ try{ return e && e.offsetParent !== null && e.clientHeight>0 && e.clientWidth>0; }catch(e){return false;} }
  var keywords = ['写评论','发表评论','说点什么','输入评论','回复'];
  var els = Array.from(document.querySelectorAll('[contenteditable], textarea, [role="textbox"], input'));
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
    if(el.tagName && (el.tagName.toLowerCase()==='textarea' || el.tagName.toLowerCase()==='input')){
      el.focus(); el.value = text; el.dispatchEvent(new Event('input',{bubbles:true}));
    } else {
      el.focus(); el.innerText = text; el.dispatchEvent(new Event('input',{bubbles:true}));
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

def send_comment(driver, text, dry_run=True, wait_after=1.2):
    try:
        els = driver.find_elements(By.ID, COMMENT_BOX_ID)
        for el in els:
            try:
                if not el.is_displayed():
                    continue
            except Exception:
                continue

            try:
                driver.execute_script("arguments[0].innerHTML = arguments[1]; arguments[0].dispatchEvent(new Event('input',{bubbles:true}));", el, text)
            except Exception:
                try:
                    driver.execute_script("arguments[0].innerText = arguments[1]; arguments[0].dispatchEvent(new Event('input',{bubbles:true}));", el, text)
                except Exception:
                    logging.debug("填入评论失败（innerHTML/innerText 双重尝试均失败）")
            time.sleep(0.5 + random.random() * 0.8)

            try:
                send_btn = driver.find_element(By.CSS_SELECTOR, SEND_BTN_CSS)
                if dry_run:
                    return True, "dry_run_send_available"
                send_btn.click()
                time.sleep(wait_after + random.random() * 0.8)
                return True, "sent_by_button"
            except Exception:
                try:
                    res = driver.execute_script(_JS_CLICK_SEND)
                    if isinstance(res, dict) and res.get('ok'):
                        return True, "sent_by_js_click"
                except Exception:
                    pass
                try:
                    el.send_keys(Keys.ENTER)
                    time.sleep(wait_after + random.random() * 0.8)
                    return True, "sent_by_enter"
                except Exception:
                    return False, "no_send_method"
        try:
            res = driver.execute_script(_JS_FILL, text)
            if not (isinstance(res, dict) and res.get('ok')):
                return False, "js_fill_failed"
        except Exception as e:
            logging.debug("执行 _JS_FILL 异常：%s", e)
            return False, "js_fill_exception"

        try:
            res2 = driver.execute_script(_JS_CLICK_SEND)
            if isinstance(res2, dict) and res2.get('ok'):
                return True, "sent_by_js"
            else:
                return False, "js_click_failed"
        except Exception as e:
            logging.debug("执行 _JS_CLICK_SEND 异常：%s", e)
            return False, "js_click_exception"
    except Exception as e:
        logging.exception("send_comment 异常：%s", e)
        return False, "exception"

def post_actions_on_note(driver, note_url, do_like=False, do_collect=False, comment_text=None, dry_run=True, skip_verification=False, refresh_retries=4, hide_overlay_if_fail=True):
    logging.info("打开笔记：%s", note_url)
    try:
        driver.get(note_url)
    except Exception as e:
        logging.warning("打开页面异常：%s", e)
        return False, "open_failed"
    time.sleep(1.5 + random.random() * 1.2)

    viewable = ensure_note_viewable(driver, max_retries=refresh_retries, hide_overlay_if_fail=hide_overlay_if_fail)
    if not viewable:
        logging.warning("页面不可浏览，跳过：%s", note_url)
        return False, "not_viewable"

    try:
        has_box = any((el.is_displayed() for el in driver.find_elements(By.ID, COMMENT_BOX_ID)))
    except Exception:
        has_box = False
    try:
        has_send = any((b.is_displayed() for b in driver.find_elements(By.CSS_SELECTOR, SEND_BTN_CSS)))
    except Exception:
        has_send = False

    if not (has_box or has_send) and not skip_verification:
        keywords = ['验证码', '请验证', '安全验证', '访问受限', '请登录']
        seen = False
        for kw in keywords:
            try:
                elems = driver.find_elements(By.XPATH, f"//*[contains(normalize-space(.), '{kw}')]")
                for e in elems:
                    try:
                        if e.is_displayed():
                            seen = True
                            break
                    except Exception:
                        continue
                if seen:
                    break
            except Exception:
                continue
        if seen:
            logging.warning("检测到可见的验证提示，请人工处理或使用 --skip-verification")
            input("请在浏览器中完成验证或登录，完成后回车继续...")

    if do_like:
        try:
            ok_like, msg_like = like_wrapper_action(driver, dry_run=dry_run)
            logging.info("点赞结果：%s, %s", ok_like, msg_like)
        except Exception as e:
            logging.exception("点赞异常：%s", e)

    if do_collect:
        try:
            ok_col, msg_col = collect_wrapper_action(driver, dry_run=dry_run)
            logging.info("收藏结果：%s, %s", ok_col, msg_col)
        except Exception as e:
            logging.exception("收藏异常：%s", e)

    if comment_text:
        try:
            ok_c, msg_c = send_comment(driver, comment_text, dry_run=dry_run)
            logging.info("评论结果：%s, %s", ok_c, msg_c)
            return ok_c, msg_c
        except Exception as e:
            logging.exception("评论异常：%s", e)
            return False, "comment_exception"

    return True, "done_no_comment"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--url-col", type=str, default=URL_COL)
    parser.add_argument("--comments-file", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="不实际点击")
    parser.add_argument("--do-like", action="store_true", default=True, help="点赞")
    parser.add_argument("--do-collect", action="store_true", default=True, help="收藏")
    parser.add_argument("--no-comment", action="store_true", help="评论")
    parser.add_argument("--min-delay", type=float, default=MIN_DELAY)
    parser.add_argument("--max-delay", type=float, default=MAX_DELAY)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--user-data-dir", type=str, default=None)
    parser.add_argument("--driver-path", type=str, default=None)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--refresh-retries", type=int, default=4)
    parser.add_argument("--no-overlay-hide", action="store_true", help="overlay hide")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logging.error("找不到 CSV 文件：%s", args.csv)
        return

    comments_list = DEFAULT_COMMENTS.copy()
    if args.comments_file and os.path.exists(args.comments_file):
        with open(args.comments_file, encoding='utf-8') as cf:
            lines = [l.strip() for l in cf if l.strip()]
            if lines:
                comments_list = lines

    posted = load_posted_log(POSTED_LOG)
    rows = load_csv_links(args.csv, url_col=args.url_col)
    logging.info("准备处理 %d 条候选（已排除 %d 条已处理）", len(rows), sum(1 for r in rows if r['url'] in posted))

    try:
        driver = create_driver(headless=args.headless, user_data_dir=args.user_data_dir, driver_path=args.driver_path)
    except Exception as e:
        logging.error("创建 WebDriver 失败：%s", e)
        if not _WDM_AVAILABLE and not args.driver_path:
            logging.error("未检测到 webdriver-manager，且未指定 --driver-path。")
        return

    try:
        driver.get("https://www.xiaohongshu.com")
        logging.info("登录")
        input("回车继续")

        counter = 0
        for item in rows:
            url = item['url']
            if url in posted:
                logging.info("已处理，跳过：%s", url)
                continue
            counter += 1
            comment_text = None if args.no_comment else random_comment(comments_list)

            success = False
            tries = 0
            while tries < MAX_TRIES_PER_POST and not success:
                tries += 1
                try:
                    ok, reason = post_actions_on_note(
                        driver,
                        url,
                        do_like=args.do_like,
                        do_collect=args.do_collect,
                        comment_text=comment_text,
                        dry_run=args.dry_run,
                        skip_verification=args.skip_verification,
                        refresh_retries=args.refresh_retries,
                        hide_overlay_if_fail=(not args.no_overlay_hide)
                    )
                    if ok:
                        success = True
                        logging.info("处理完成（%s）: %s", reason, url)
                        posted.add(url)
                        append_posted_log(POSTED_LOG, url if not args.dry_run else (url + " [dryrun]"))
                        break
                    else:
                        logging.warning("尝试失败 (%s)，重试 %d/%d", reason, tries, MAX_TRIES_PER_POST)
                        time.sleep(2 + random.random() * 2)
                except Exception as e:
                    logging.exception("处理该条目时异常：%s", e)
                    time.sleep(2 + random.random() * 2)

            delay = random.uniform(max(0.1, args.min_delay), max(0.1, args.max_delay))
            logging.info("等待 %.1f 秒后继续...", delay)
            time.sleep(delay)

            if counter % LONG_BREAK_EVERY == 0:
                lb = random.uniform(LONG_BREAK_MIN, LONG_BREAK_MAX)
                logging.info("已处理 %d 条，长休息 %.1f 秒", counter, lb)
                time.sleep(lb)

        logging.info("全部处理完毕")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()
