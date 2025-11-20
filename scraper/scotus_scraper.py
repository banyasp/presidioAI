import re
import io
import os
import json
import sqlite3
import argparse
import logging
from pathlib import Path
import sys
from urllib.parse import urlparse
from datetime import datetime

# Ensure the repo root and scraper dir are on sys.path so utils imports resolve
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from case_mention_extractor import extract_case_mentions_from_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("scotus")

# Default index URLs to scrape: years 24 down to 17 (inclusive).
# This covers the slip opinion index pages for those Court terms.
DEFAULT_INDEX_URLS = [
    f"https://www.supremecourt.gov/opinions/slipopinion/{y}"
    for y in range(24, 16, -1)
]
BASE = "https://www.supremecourt.gov"

# ------------------------------
# Scrape and parse helpers
# ------------------------------

def fetch_index(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def _normalize(s):
    return re.sub(r"\s+", " ", s or "").strip()

def _build_colmap(tr):
    headers = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
    colmap = {}
    for i, h in enumerate(headers):
        hlow = h.lower()
        if "date" in hlow and "update" not in hlow:
            colmap["date"] = i
        elif "docket" in hlow:
            colmap["docket"] = i
        elif hlow in {"name", "case name"} or "name" in hlow:
            colmap["name"] = i
    return colmap

def parse_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue
        colmap = _build_colmap(trs[0])
        if not {"date", "docket", "name"}.issubset(colmap):
            continue

        for tr in trs[1:]:
            tds = tr.find_all("td")
            if len(tds) <= max(colmap.values()):
                continue

            date = _normalize(tds[colmap["date"]].get_text(" ", strip=True))
            docket = _normalize(tds[colmap["docket"]].get_text(" ", strip=True))
            case_name_cell = tds[colmap["name"]]
            case_name = _normalize(case_name_cell.get_text(" ", strip=True))

            pdf_url = None
            for a in tr.select('a[href]'):
                href = a.get("href", "")
                if href.lower().endswith(".pdf"):
                    pdf_url = href if href.startswith("http") else BASE + href
                    break
            if not pdf_url:
                continue

            results.append({
                "case_name": case_name,
                "docket": docket,
                "date": date,
                "pdf_url": pdf_url
            })
    return results

def _span_is_italic(span):
    if "italic" in span:
        return bool(span["italic"])
    return bool(span.get("flags", 0) & 2)

def _get_page_text(doc, pno):
    return doc.load_page(pno).get_text("text")

def _get_page_dict(doc, pno):
    return doc.load_page(pno).get_text("dict")

def _find_first_syllabus_page(doc, max_pages=5):
    upto = min(max_pages, doc.page_count)
    for p in range(upto):
        txt = _get_page_text(doc, p)
        m = re.search(r"\bSyllabus\b", txt, flags=re.I)
        if m:
            return p, m.start()
    return None, None

def _find_italic_held_location(doc, start_page):
    joined = []
    offsets = []
    total = 0
    for p in range(start_page, doc.page_count):
        t = _get_page_text(doc, p)
        offsets.append(total)
        joined.append(t)
        total += len(t)
    all_text = "".join(joined)  # noqa: F841

    for rel_idx, p in enumerate(range(start_page, doc.page_count)):
        pd = _get_page_dict(doc, p)
        for b in pd.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    s_text = s.get("text", "")
                    if not s_text:
                        continue
                    s_norm = re.sub(r"\s+", " ", s_text)
                    if re.match(r"^Held\b[:]?$", s_norm) and _span_is_italic(s):
                        page_text = _get_page_text(doc, p)
                        m = re.search(r"\bHeld\b[:]?[\s]*", page_text)
                        if m:
                            global_idx = offsets[rel_idx] + m.start()
                            return start_page + rel_idx, global_idx, s_text
    return None, None, None

def clean_page_proofs(text):
    """Remove content between 'Page Proof Pending Publication' and the next 'Syllabus' or 'Opinion of the Court'."""
    pattern = r'Page Proof Pending Publication.*?(?=(?:Syllabus|Opinion of the Court))'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def _clean(s):
    s = clean_page_proofs(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()

def extract_structured_from_pdf(pdf_bytes, title_hint=None):
    doc = fitz.open(stream=io.BytesIO(pdf_bytes).getvalue(), filetype="pdf")

    syl_page, _ = _find_first_syllabus_page(doc, max_pages=5)
    if syl_page is None:
        return None

    pages_text = [_get_page_text(doc, p) for p in range(syl_page, doc.page_count)]
    joined_text = "".join(pages_text)

    m_syl = re.search(r"\bSyllabus\b", joined_text, flags=re.I)
    if not m_syl:
        return None
    syl_start = m_syl.start()

    held_page, held_idx, _ = _find_italic_held_location(doc, syl_page)
    if held_page is None or held_idx is None:
        return None

    window_lo = max(0, held_idx - 200)
    window_hi = min(len(joined_text), held_idx + 200)
    window = joined_text[window_lo:window_hi]
    m_held = re.search(r"\bHeld\s*:\s*", window)
    if not m_held:
        return None

    held_abs_start = window_lo + m_held.start()
    held_abs_end = window_lo + m_held.end()

    case_facts = _clean(joined_text[syl_start:held_abs_start])

    after_held = joined_text[held_abs_end:]
    m_op = re.search(r"delivered\s+the\s+opinion\s+of\s+the\s+Court", after_held, flags=re.I)

    region = after_held[:m_op.start()] if m_op else after_held
    paras = [p.strip() for p in re.split(r"\n\s*\n", region) if p.strip()]
    decision = _clean(paras[0]) if paras else ""

    opinion_tail = after_held[m_op.end():] if m_op else ""
    case_mentions = []
    related_cases = []
    if opinion_tail:
        LOG.info("Passing opinion text to case_mention_extractor for %s", title_hint or "")
        case_mentions = extract_case_mentions_from_text(opinion_tail)
        LOG.info("Found %d case mention(s) in %s: %s", len(case_mentions), title_hint or "", case_mentions)
        related_cases = case_mentions
    else:
        LOG.info("No opinion tail found after 'delivered the opinion of the Court' for %s", title_hint or "")

    if not case_facts or not decision:
        return None

    return {
        "title": title_hint or "",
        "case_facts": case_facts,
        "decision": decision,
        "case_mentions": case_mentions,
        "related_cases": related_cases,
    }

# ------------------------------
# Database helpers
# ------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY,
    case_name TEXT NOT NULL,
    date TEXT,
    docket_number TEXT,
    case_facts TEXT,
    decision TEXT,
    related_cases TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(docket_number, date)
);
"""

def init_db(db_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(DDL)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(cases)")}
        if "related_cases" not in cols:
            conn.execute("ALTER TABLE cases ADD COLUMN related_cases TEXT;")
        conn.commit()
    LOG.info("Database ready at %s", db_path)

def upsert_case(conn, row):
    sql = """
    INSERT INTO cases (case_name, date, docket_number, case_facts, decision, related_cases)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(docket_number, date) DO UPDATE SET
        case_name=excluded.case_name,
        case_facts=excluded.case_facts,
        decision=excluded.decision,
        related_cases=excluded.related_cases
    """
    vals = (
        row["case_name"],
        row["date"],
        row["docket"],
        row["case_facts"],
        row["decision"],
        row.get("related_cases", ""),
    )
    conn.execute(sql, vals)

def save_cases(db_path: str, items):
    count = 0
    with sqlite3.connect(db_path) as conn:
        for it in items:
            try:
                upsert_case(conn, it)
                count += 1
            except Exception as e:
                LOG.warning("Skip row %s due to %s", it.get("docket", "?"), e)
        conn.commit()
    LOG.info("Saved %d rows", count)

# ------------------------------
# Orchestration
# ------------------------------

def _index_year(url: str) -> str:
    """Return the slip opinion year suffix (e.g., '24') from the index URL."""
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1]

def _parse_year_limits(spec: str):
    """Parse comma-separated YEAR=NUM pairs into a dict like {'24': 5, '23': 2}."""
    limits = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"Invalid year limit '{part}', expected YEAR=NUM")
        year, num = part.split("=", 1)
        year = year.strip()
        try:
            limits[year] = int(num)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid number in year limit '{part}'")
    return limits

def scrape_and_extract(index_url: str, limit: int | None = None):
    html = fetch_index(index_url)
    rows = parse_rows(html)
    LOG.info("Found %d case rows in %s", len(rows), index_url)
    if limit is not None:
        rows = rows[:limit]
        LOG.info("Applying limit %s, processing %d case(s): %s", limit, len(rows), [(r['case_name'], r['docket']) for r in rows])
    else:
        LOG.info("Processing all %d case(s): %s", len(rows), [(r['case_name'], r['docket']) for r in rows])

    extracted = []
    for r in rows:
        try:
            LOG.info("Fetching PDF for %s (%s)", r["case_name"], r["docket"])
            resp = requests.get(r["pdf_url"], timeout=60)
            resp.raise_for_status()
            parsed = extract_structured_from_pdf(resp.content, title_hint=r["case_name"])
            if not parsed:
                LOG.info("No structured extract for %s", r["case_name"])
                continue
            related_cases_str = json.dumps(parsed.get("related_cases", []))
            extracted.append({
                "case_name": parsed["title"] or r["case_name"],
                "docket": r["docket"],
                "date": r["date"],
                "case_facts": parsed["case_facts"],
                "decision": parsed["decision"],
                "related_cases": related_cases_str,
            })
            if parsed.get("case_mentions"):
                LOG.info("Case mentions passed along for %s: %s", r["case_name"], parsed["case_mentions"])
            else:
                LOG.info("No case mentions extracted for %s", r["case_name"])
        except Exception as e:
            LOG.warning("Error processing %s: %s", r.get("pdf_url", "unknown"), e)
    return extracted

def main():
    ap = argparse.ArgumentParser(description="SCOTUS slip opinions to SQLite")
    ap.add_argument("--db", default="./scotus_cases.db", help="SQLite file path")
    ap.add_argument(
        "--index-urls",
        nargs="+",
        default=DEFAULT_INDEX_URLS,
        help="One or more slip opinion index URLs, e.g. .../24 .../23",
    )
    ap.add_argument(
        "--per-index-limit",
        type=int,
        default=None,
        help="Max cases to process per index URL (omit for no limit; overridden by per-year limits)",
    )
    ap.add_argument(
        "--per-year-limit",
        type=_parse_year_limits,
        default=None,
        help="Comma list of YEAR=NUM caps (e.g., 24=5,23=3) to limit cases per slip opinion year",
    )
    args = ap.parse_args()

    init_db(args.db)

    all_items = []
    for url in args.index_urls:
        year_key = _index_year(url)
        year_limit = None
        if args.per_year_limit:
            year_limit = args.per_year_limit.get(year_key)
        limit = year_limit if year_limit is not None else args.per_index_limit
        LOG.info("Scraping index %s (year=%s, limit=%s)", url, year_key, limit or "all")
        items = scrape_and_extract(url, limit=limit)
        all_items.extend(items)

    save_cases(args.db, all_items)
    LOG.info("Inserted or updated %d cases total", len(all_items))
    LOG.info("Done at %s", datetime.now().isoformat(timespec="seconds"))

if __name__ == "__main__":
    main()
