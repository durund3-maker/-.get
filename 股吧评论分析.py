
from __future__ import annotations
import sys
import argparse
import glob
import io
import logging
import os
import re
import gc
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT = r"E:\2025.8.22 股吧评论\数据"
DEFAULT_OUTPUT = r"E:\2025.8.22 股吧评论\输出"
DEFAULT_HOT_THRESHOLD = 100

REQUIRED_COLS = [
    "user_id", "post_id", "post_source_id", "post_type", "user_name",
    "post_publish_time", "stockbar_name", "stockbar_code", "forward",
    "coment_count", "click_count", "post_title", "url", "content"
]


LEXICON_SYMPATHY = [
    "同情", "深表同情", "同情一下", "怜悯", "恻隐之心", "可怜", "真可怜", "太可怜了",
    "心疼", "真心疼", "太心疼了", "惋惜", "太惋惜了", "深感惋惜", "唏嘘", "令人唏嘘",
    "唏嘘不已", "悲悯", "悲悯之心", "不忍", "实在不忍", "不忍直视", "怜惜", "令人怜惜",
    "理解", "可以理解", "能够理解", "理解这种处境", "体谅", "互相体谅", "可以体谅",
    "共情", "产生共情", "共情了", "同感", "深有同感", "有同感", "换位思考", "换个角度想",
    "设身处地", "设身处地想的话", "感同身受", "能感同身受", "懂", "我懂", "我懂的",
    "心酸", "太心酸了", "感到心酸", "揪心", "真揪心", "太揪心了", "难过", "挺难过的",
    "心里难过", "伤心", "有点伤心", "心里不是滋味", "不是滋味", "心寒", "悲哀", "感到悲哀",
    "哀叹", "令人哀叹", "不容易啊", "太不容易了", "也不容易", "太难了", "太难了这情况",
    "太惨了", "这情况太惨了", "处境艰难", "不公平", "太不公平了", "不公平待遇", "过分",
    "太过分了", "冤枉", "被冤枉", "太冤了", "无辜", "太无辜了", "倒霉", "太倒霉了",
    "替它不值", "替企业不值",
    "挺住", "支持国产", "支持国货",
    "民族企业的脊梁",  "民族企业", "国人当自强", "自强",  "美国打压", "西方制裁", "卡脖子",
    "为国", "牺牲", "接盘", "憋屈", "太憋屈了",
    "咽不下这口气", "争口气",  "团结", "共渡难关", "多难兴邦",
    "自研不易", "自主创新", "打破垄断", "逆势而上", "负重前行", "致敬",
    "栋梁", "骄傲", "欺负", "针对", "恶意做空","争点气"
]

LEXICON_DISGUST = [
    "厌恶", "恶心", "反感", "作呕", "讨厌", "憎恶", "痛恨", "鄙视", "呕心",
    "垃圾", "烂", "离谱", "无语", "辣眼睛", "倒胃口", "厌烦", "讨厌至极",
    "害怕", "怕了", "好怕", "太怕了", "恐惧", "恐惧感", "惊恐", "惊吓", "吓到了",
    "担忧", "很担忧", "非常担忧", "忧心忡忡", "担心", "好担心", "太担心了", "担惊受怕",
    "恐慌", "有点恐慌", "恐慌了", "恐慌不已", "焦虑", "很焦虑", "焦虑不安", "焦躁",
    "后怕", "有点后怕", "心有余悸", "忐忑", "忐忑不安", "惴惴不安", "惶惶不安",
    "怕出事", "怕不行了", "怕撑不住", "怕垮掉", "怕完蛋", "怕跌", "怕暴跌", "怕崩盘",
    "吓破胆", "胆战心惊", "毛骨悚然", "不寒而栗",
    "失望", "很失望", "太失望了", "失望透顶", "大失所望", "彻底失望", "满心失望",
    "寒心", "太寒心了", "令人寒心", "心寒透了", "绝望", "有点绝望", "太绝望了", "绝望透顶",
    "死心", "彻底死心", "心死", "心灰意冷", "灰心", "灰心丧气", "没指望", "看不到希望",
    "白期待了", "期望落空", "指望不上", "令人失望", "太让人失望了", "空欢喜", "白费功夫",
    "失落", "极度失落", "沮丧", "非常沮丧", "泄气", "提不起劲", "意难平",
    "膈应", "让人膈应", "看不惯", "真看不惯", "唾弃", "令人唾弃", "嫌恶", "招人嫌"
]

def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(out_dir: Path):
    log_file = out_dir / "processing.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    )
    logging.info("日志写入：%s", str(log_file))

def list_input_files(input_path: str | Path) -> List[Path]:
    p = Path(input_path)
    if p.is_dir():
        files = sorted(Path(f) for f in glob.glob(str(p / "*.csv")))
    elif p.is_file():
        files = [p]
    else:
        files = sorted(Path(f) for f in glob.glob(str(p)))
    if not files:
        raise FileNotFoundError(f"未在 {input_path} 找到任何 CSV 文件")
    return files

def parse_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    mask = dt.isna() & s.notna()
    if mask.any():
        s2 = s[mask].astype(str).str.replace("-", "/").str.replace(".", "/")
        dt2 = pd.to_datetime(s2, errors="coerce")
        dt.loc[mask] = dt2
    return dt

def clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        txt = ""
    t = txt.strip()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"www\.\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[@#][\w\-\._一-龥]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def build_lex_pattern(words: List[str]) -> Optional[re.Pattern]:
    ws = [w for w in words if isinstance(w, str) and w.strip()]
    if not ws:
        return None
    esc = sorted({re.escape(w) for w in ws}, key=len, reverse=True)
    pat = "|".join(esc)
    try:
        return re.compile(pat)
    except Exception:
        return None

def emotion_scores_for_series_fast(s: pd.Series,
                                   sym_pat: Optional[re.Pattern],
                                   dis_pat: Optional[re.Pattern],
                                   lex_sym: List[str], lex_dis: List[str]) -> pd.DataFrame:
    cleaned = s.fillna("").astype(str).map(clean_text)
    if sym_pat is not None:
        sym_hits = cleaned.str.count(sym_pat).fillna(0).astype(int).to_numpy()
    else:
        sym_hits = np.zeros(len(cleaned), dtype=np.int32)
        for w in lex_sym:
            if not w: continue
            sym_hits += cleaned.str.count(re.escape(w)).fillna(0).astype(int).to_numpy()

    if dis_pat is not None:
        dis_hits = cleaned.str.count(dis_pat).fillna(0).astype(int).to_numpy()
    else:
        dis_hits = np.zeros(len(cleaned), dtype=np.int32)
        for w in lex_dis:
            if not w: continue
            dis_hits += cleaned.str.count(re.escape(w)).fillna(0).astype(int).to_numpy()

    effective_chars = cleaned.str.count(r"[A-Za-z0-9\u4e00-\u9fa5]").fillna(0).astype(int).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sym_index = np.where(effective_chars > 0, sym_hits / effective_chars * 1000.0, 0.0)
        dis_index = np.where(effective_chars > 0, dis_hits / effective_chars * 1000.0, 0.0)

    return pd.DataFrame({
        "sympathy_hits": sym_hits,
        "disgust_hits": dis_hits,
        "sympathy_index": sym_index,
        "disgust_index": dis_index,
        "effective_chars": effective_chars
    })

def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        # A: 尝试 C 引擎（更快），跳过坏行
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception as e:
            last_err = e
        # B: 尝试 python 引擎
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception as e:
            last_err = e
        # C: open(..., errors='replace') 然后交给 pandas
        try:
            with open(path, "r", encoding=enc, errors="replace") as fh:
                return pd.read_csv(fh)
        except Exception as e:
            last_err = e
        # D: chunks
        try:
            chunks = pd.read_csv(path, encoding=enc, engine="python", chunksize=100_000)
            return pd.concat(chunks, ignore_index=True)
        except Exception as e:
            last_err = e
        gc.collect()
    raise RuntimeError(f"读取 {path} 失败: {last_err}")

def quarter_month_collapse(m: int) -> int:
    if m in (1,2,3): return 3
    if m in (4,5,6): return 6
    if m in (7,8,9): return 9
    return 12

def make_yq(dt: pd.Series) -> Tuple[pd.Series, pd.Series]:
    year = dt.dt.year.astype("Int64")
    month = dt.dt.month.astype("Int64")
    month_q = month.map(quarter_month_collapse).astype("Int64")
    q_map = {3:1, 6:2, 9:3, 12:4}
    quarter = month_q.map(q_map).astype("Int64")
    yq = year.astype(str) + "q" + quarter.astype(str)
    return month_q, yq


def get_per_post_minimal(df_aug: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "post_id", "user_id", "post_datetime", "stockbar_code",
        "coment_count", "is_hot",
        "sympathy_hits", "disgust_hits",
        "sympathy_index", "disgust_index",
        "effective_chars", "source_file"
    ]
    for c in keep:
        if c not in df_aug.columns:
            df_aug[c] = pd.NA
    return df_aug[keep].copy()

def write_part(df: pd.DataFrame, parts_dir: Path, base_name: str,
               fmt: str, idx: int) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        path = parts_dir / f"{idx:05d}_{base_name}.parquet"
        try:
            df.to_parquet(path, index=False, compression="snappy")
            return path
        except Exception as e:
            logging.warning("写 Parquet 失败（退回 CSV.GZ）：%s", e)
            path = parts_dir / f"{idx:05d}_{base_name}.csv.gz"
            df.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
            return path
    else:
        path = parts_dir / f"{idx:05d}_{base_name}.csv.gz"
        df.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        return path

def stem_processed(parts_dir: Path, stem: str) -> bool:
    if not parts_dir.exists():
        return False
    for ext in ("*.parquet", "*.csv.gz"):
        if any(parts_dir.glob(f"{stem}*.parquet")) or any(parts_dir.glob(f"{stem}*.csv.gz")):
            return True
    return False


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("CREATE TABLE IF NOT EXISTS seen_posts (post_id TEXT PRIMARY KEY)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_users (
            g TEXT NOT NULL,
            user_id TEXT NOT NULL,
            PRIMARY KEY (g, user_id)
        )
    """)
    conn.commit()
    return conn

def upsert_seen_posts(conn: sqlite3.Connection, post_ids: List[str]) -> List[bool]:
    cur = conn.cursor()
    BATCH = 5000
    mask = []
    for i in range(0, len(post_ids), BATCH):
        sub = post_ids[i:i+BATCH]
        q = "SELECT post_id FROM seen_posts WHERE post_id IN ({})".format(",".join("?"*len(sub)))
        exist = set(r[0] for r in cur.execute(q, sub).fetchall())
        sub_mask = []
        new_vals = []
        for pid in sub:
            if not pid:
                sub_mask.append(False)
            elif pid in exist:
                sub_mask.append(False)
            else:
                sub_mask.append(True)
                new_vals.append((pid,))
        if new_vals:
            cur.executemany("INSERT OR IGNORE INTO seen_posts(post_id) VALUES (?)", new_vals)
        mask.extend(sub_mask)
    conn.commit()
    return mask

def upsert_seen_users(conn: sqlite3.Connection, groups: List[str], user_ids: List[str]):
    cur = conn.cursor()
    BATCH = 5000
    for i in range(0, len(groups), BATCH):
        vals = [(groups[j], user_ids[j]) for j in range(i, min(i+BATCH, len(groups))) if groups[j] and user_ids[j]]
        if vals:
            cur.executemany("INSERT OR IGNORE INTO seen_users(g, user_id) VALUES (?,?)", vals)
    conn.commit()


def main():

    if len(sys.argv) <= 1 or sys.argv[1] not in ("process", "build-panel"):
        sys.argv.insert(1, "process")

    parser = argparse.ArgumentParser(description="股吧情绪分析 —— 快速续跑 & 面板构建")
    sub = parser.add_subparsers(dest="mode")

    p1 = sub.add_parser("process", help="处理原始CSV生成 per_post 分片（快速续跑）")
    p1.add_argument("-i", "--input", type=str, default=DEFAULT_INPUT)
    p1.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT)
    p1.add_argument("-t", "--threshold", type=int, default=DEFAULT_HOT_THRESHOLD)
    p1.add_argument("--sym-lex", type=str, default=None)
    p1.add_argument("--dis-lex", type=str, default=None)
    p1.add_argument("--per-post", choices=["off", "min", "full"], default="min")
    p1.add_argument("--per-post-format", choices=["parquet", "csv"], default="parquet")
    p1.add_argument("--skip-existing", action="store_true", help="已有分片则跳过该CSV")
    p1.add_argument("--sample-rate", type=float, default=1.0, help="per-post 抽样比例 (0,1]")

    p2 = sub.add_parser("build-panel", help="从 per_post 分片构建季度面板（全局去重、低内存）")
    p2.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT)
    p2.add_argument("--parts-dir", type=str, default=None, help="per_post 分片目录（默认 output/per_post_parts）")

    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output if hasattr(args, "output") else DEFAULT_OUTPUT)
    setup_logging(output_dir)

    if args.mode == "process":
        lex_sym = LEXICON_SYMPATHY
        lex_dis = LEXICON_DISGUST
        if args.sym_lex:
            try:
                with open(args.sym_lex, "r", encoding="utf-8") as fh:
                    lex_sym = [ln.strip() for ln in fh if ln.strip()]
                logging.info("加载同情词典：%s（%d 词）", args.sym_lex, len(lex_sym))
            except Exception as e:
                logging.warning("同情词典加载失败，使用内置：%s", e)
        if args.dis_lex:
            try:
                with open(args.dis_lex, "r", encoding="utf-8") as fh:
                    lex_dis = [ln.strip() for ln in fh if ln.strip()]
                logging.info("加载厌恶词典：%s（%d 词）", args.dis_lex, len(lex_dis))
            except Exception as e:
                logging.warning("厌恶词典加载失败，使用内置：%s", e)

        sym_pat = build_lex_pattern(lex_sym)
        dis_pat = build_lex_pattern(lex_dis)

        if args.per_post != "off" and args.per_post_format == "parquet":
            try:
                import pyarrow  # noqa
            except Exception:
                logging.warning("未检测到 pyarrow，per_post 分片将使用 CSV.GZ。pip install pyarrow 可更快。")
                args.per_post_format = "csv"

        parts_dir = output_dir / "per_post_parts"
        manifest_path = output_dir / "per_post_manifest.csv"
        if not manifest_path.exists():
            with open(manifest_path, "w", encoding="utf-8") as fh:
                fh.write("part,rows,bytes\n")

        files = list_input_files(args.input)
        logging.info("发现 %d 个文件，开始处理（不做在线聚合，避免后期变慢）", len(files))

        for idx, f in enumerate(tqdm(files, desc="读取CSV", unit="file"), start=1):
            stem = Path(f).stem
            try:
                if args.skip_existing and stem_processed(parts_dir, stem):
                    logging.info("跳过已处理文件：%s", f)
                    continue

                df = safe_read_csv(Path(f))

                # 补齐列
                for col in REQUIRED_COLS:
                    if col not in df.columns:
                        df[col] = pd.NA
                df = df[REQUIRED_COLS].copy()
                df["source_file"] = str(f)

                for col in ["forward", "coment_count", "click_count"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["post_datetime"] = parse_datetime(df["post_publish_time"])
                df["stockbar_code"] = (df["stockbar_code"]
                                       .fillna("").astype(str).str.strip()
                                       .str.upper().str.replace(r"[^A-Z0-9]", "", regex=True))
                df.loc[df["stockbar_code"] == "", "stockbar_code"] = pd.NA

                emo = emotion_scores_for_series_fast(df["content"], sym_pat, dis_pat, lex_sym, lex_dis)
                df = pd.concat([df.reset_index(drop=True), emo.reset_index(drop=True)], axis=1)

                df["is_hot"] = (df["coment_count"] >= args.threshold).astype(int)

                if args.per_post != "off":
                    out_df = df if args.per_post == "full" else get_per_post_minimal(df)

                    if args.sample_rate < 0.9999:
                        ids = out_df["post_id"].fillna("").astype(str)
                        hk = ids.str.encode("utf-8").map(lambda b: (hash(b) & 0xffffffff) / 0xffffffff)
                        out_df = out_df.loc[hk < max(1e-6, float(args.sample_rate))].reset_index(drop=True)

                    part_path = write_part(out_df, parts_dir, stem, args.per_post_format, idx)
                    size = part_path.stat().st_size if part_path.exists() else 0
                    with open(manifest_path, "a", encoding="utf-8") as fh:
                        fh.write(f"{part_path},{len(out_df)},{size}\n")
                    logging.info("分片写出：%s 行=%d 体积=%.2fMB", part_path.name, len(out_df), size/1024/1024)

                del df, emo
                gc.collect()

            except Exception as e:
                logging.warning("跳过文件 %s（处理失败）：%s", str(f), str(e))
                continue

        logging.info("process 模式完成。分片目录：%s", str(parts_dir))
        return

    if args.mode == "build-panel":
        parts_dir = Path(args.parts_dir) if args.parts_dir else (output_dir / "per_post_parts")
        if not parts_dir.exists():
            raise RuntimeError(f"未找到分片目录：{parts_dir}")

        panel_rows: Dict[Tuple[str,int,int], Dict[str,Any]] = {}

        def add_row(key, sym_idx, dis_idx, sym_hits, dis_hits, eff_chars, is_hot):
            rec = panel_rows.get(key)
            if rec is None:
                rec = {
                    "posts_n": 0,
                    "sym_sum": 0.0,
                    "dis_sum": 0.0,
                    "sym_hits_sum": 0,
                    "dis_hits_sum": 0,
                    "effective_chars_sum": 0,
                    "hot_posts_n": 0
                }
                panel_rows[key] = rec
            rec["posts_n"] += 1
            rec["sym_sum"] += float(sym_idx or 0.0)
            rec["dis_sum"] += float(dis_idx or 0.0)
            rec["sym_hits_sum"] += int(sym_hits or 0)
            rec["dis_hits_sum"] += int(dis_hits or 0)
            rec["effective_chars_sum"] += int(eff_chars or 0)
            rec["hot_posts_n"] += int(is_hot or 0)

        db_path = output_dir / "dedupe.sqlite"
        conn = open_db(db_path)

        part_files = sorted(list((parts_dir).glob("*.parquet")) + list((parts_dir).glob("*.csv.gz")))
        if not part_files:
            raise RuntimeError(f"分片目录为空：{parts_dir}")
        logging.info("开始从分片构建面板（%d 个分片）...", len(part_files))

        for p in tqdm(part_files, desc="扫描分片", unit="part"):
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p, columns=[
                        "post_id", "user_id", "post_datetime", "stockbar_code",
                        "is_hot", "sympathy_hits", "disgust_hits",
                        "sympathy_index", "disgust_index", "effective_chars"
                    ])
                else:
                    df = pd.read_csv(p, usecols=[
                        "post_id", "user_id", "post_datetime", "stockbar_code",
                        "is_hot", "sympathy_hits", "disgust_hits",
                        "sympathy_index", "disgust_index", "effective_chars"
                    ])
                df["post_datetime"] = parse_datetime(df["post_datetime"])
                df["stockbar_code"] = (df["stockbar_code"].fillna("").astype(str).str.strip()
                                       .str.upper().str.replace(r"[^A-Z0-9]", "", regex=True))
                df.loc[df["stockbar_code"] == "", "stockbar_code"] = pd.NA
                df = df.dropna(subset=["post_id", "post_datetime", "stockbar_code"])
                if df.empty:
                    continue

                post_ids = df["post_id"].astype(str).tolist()
                new_mask = upsert_seen_posts(conn, post_ids)
                if not any(new_mask):
                    continue
                df = df.loc[pd.Series(new_mask, index=df.index)].reset_index(drop=True)

                month_q, _ = make_yq(df["post_datetime"])
                df["_year"] = df["post_datetime"].dt.year.astype(int)
                df["_month_q"] = month_q

                gkey = (df["stockbar_code"].astype(str) + "|" +
                        df["_year"].astype(str) + "|" +
                        df["_month_q"].astype(str))
                upsert_seen_users(conn, gkey.astype(str).tolist(), df["user_id"].fillna("").astype(str).tolist())

                for row in df.itertuples(index=False):
                    key = (getattr(row, "stockbar_code"), int(getattr(row, "_year")), int(getattr(row, "_month_q")))
                    add_row(key, getattr(row, "sympathy_index", 0.0), getattr(row, "disgust_index", 0.0),
                            getattr(row, "sympathy_hits", 0), getattr(row, "disgust_hits", 0),
                            getattr(row, "effective_chars", 0), getattr(row, "is_hot", 0))

                del df
                gc.collect()

            except Exception as e:
                logging.warning("跳过分片 %s（读取/汇总失败）：%s", str(p), str(e))
                continue

        logging.info("统计 users_n（全局去重）...")
        cur = conn.cursor()
        users_counts = {}
        for g, cnt in cur.execute("SELECT g, COUNT(*) FROM seen_users GROUP BY g"):
            try:
                stock, year, month_q = g.split("|")
                users_counts[(stock,int(year),int(month_q))] = cnt
            except Exception:
                continue

        rows = []
        q_map = {3:1,6:2,9:3,12:4}
        for (stock, year, month_q), rec in panel_rows.items():
            posts_n = rec["posts_n"]
            sym_mean = rec["sym_sum"] / posts_n if posts_n else 0.0
            dis_mean = rec["dis_sum"] / posts_n if posts_n else 0.0
            quarter = q_map.get(month_q, 4)
            yq = f"{year}q{quarter}"
            rows.append({
                "stockbar_code": stock,
                "year": year,
                "month_q": month_q,
                "quarter": quarter,
                "yq": yq,
                "posts_n": posts_n,
                "users_n": users_counts.get((stock,year,month_q), np.nan),
                "sympathy_index_mean": sym_mean,
                "disgust_index_mean": dis_mean,
                "sympathy_hits_sum": rec["sym_hits_sum"],
                "disgust_hits_sum": rec["dis_hits_sum"],
                "effective_chars_sum": rec["effective_chars_sum"],
                "hot_posts_n": rec["hot_posts_n"],
                "sympathy_index_p50": np.nan,
                "disgust_index_p50": np.nan,
            })

        panel = pd.DataFrame(rows).sort_values(["stockbar_code","year","month_q"]).reset_index(drop=True)
        panel_csv = output_dir / "panel_quarter_emotions.csv"
        panel.to_csv(panel_csv, index=False, encoding="utf-8-sig")
        logging.info("面板 CSV 已导出：%s（%d 行）", str(panel_csv), len(panel))

        panel_dta = output_dir / "panel_quarter_emotions.dta"
        try:
            panel.to_stata(panel_dta, write_index=False, version=118)
            logging.info("面板 Stata .dta 已导出：%s", str(panel_dta))
        except Exception as e:
            logging.warning("导出 Stata .dta 失败（忽略）：%s", e)

        logging.info("build-panel 完成。去重数据库：%s", str(db_path))
        return

if __name__ == "__main__":
    main()
