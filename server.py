import os, re, json, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

IST = timezone(timedelta(hours=5, minutes=30))
LEDGER_URL = os.environ["LEDGER_URL"]
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
LLM_BASE = os.environ.get("LLM_BASE", "https://aipipe.org/openai/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
STATE = {"rows": [], "rates": {}, "loaded": 0.0, "raw": 0}
LOAD_LOCK = threading.Lock()

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

# ---------------------------------------------------------------- loading
def fetch_pages(orders_url):
    first = requests.get(orders_url, timeout=30).json()
    rows, pages = list(first["orders"]), int(first.get("pages", 1))
    def get(n):
        url = re.sub(r"page=\d+", f"page={n}", orders_url)
        for _ in range(3):
            try:
                return requests.get(url, timeout=30).json()["orders"]
            except Exception:
                time.sleep(0.5)
        return []
    with ThreadPoolExecutor(8) as ex:
        for part in ex.map(get, range(2, pages + 1)):
            rows.extend(part)
    return rows

def load():
    with LOAD_LOCK:
        links = requests.get(LEDGER_URL, timeout=30).json()["links"]
        rates = requests.get(links["rates"], timeout=30).json()["usd_per_unit"]
        raw = []
        try:
            r = requests.get(links["export"], timeout=60)
            raw += [json.loads(l) for l in r.text.splitlines() if l.strip()]
        except Exception as e:
            print("export load failed:", e)
        try:
            raw += fetch_pages(links["orders"])
        except Exception as e:
            print("orders load failed:", e)
        if not raw:
            raise RuntimeError("no data loaded")
        latest = {}
        for o in raw:
            o["_upd"] = parse_ts(o["updated_at"])
            cur = latest.get(o["id"])
            if cur is None or o["_upd"] >= cur["_upd"]:
                latest[o["id"]] = o
        rows = []
        for o in latest.values():
            o["_date"] = parse_ts(o["created_at"]).astimezone(IST).date()
            o["_usd"] = Decimal(str(o["amount"])) * Decimal(str(rates[o["currency"]]))
            rows.append(o)
        STATE.update(rows=rows, rates=rates, loaded=time.time(), raw=len(raw))
        print(f"loaded {len(rows)} unique orders from {len(raw)} raw rows")

def refresher():
    while True:
        time.sleep(600)
        try:
            load()
        except Exception as e:
            print("refresh failed:", e)

def initial_load():
    while not STATE["rows"]:
        try:
            load()
        except Exception as e:
            print("initial load failed, retrying:", e)
            time.sleep(5)
    refresher()

@asynccontextmanager
async def lifespan(app):
    try:
        load()
    except Exception as e:
        print("startup load failed, retrying in background:", e)
        threading.Thread(target=initial_load, daemon=True).start()
    else:
        threading.Thread(target=refresher, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------- executor
def money(d):
    return float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def metric_value(m, rows):
    if m in ("revenue_usd", "refund_usd"):
        return money(sum((o["_usd"] for o in rows), Decimal(0)))
    if m in ("order_count", "refund_count"):
        return len(rows)
    if m == "units":
        return sum(o["qty"] for o in rows)
    if m == "distinct_customers":
        return len({o["customer"] for o in rows})
    if m == "avg_order_value_usd":
        return money(sum((o["_usd"] for o in rows), Decimal(0)) / len(rows)) if rows else 0.0
    raise ValueError(m)

def execute(spec):
    rows_all = STATE["rows"]
    m = spec.get("metric", "revenue_usd")
    if m == "list_products":
        return sorted({o["product"] for o in rows_all})
    if m == "list_regions":
        return sorted({o["region"] for o in rows_all})
    status = spec.get("status") or ("refunded" if m.startswith("refund") else "paid")
    start = date.fromisoformat(spec["start"]) if spec.get("start") else None
    end = date.fromisoformat(spec["end"]) if spec.get("end") else None
    def ok(o):
        if status != "any" and o["status"] != status: return False
        for k in ("region", "product", "customer"):
            v = spec.get(k)
            if v and o[k].lower() != str(v).lower(): return False
        if start and o["_date"] < start: return False
        if end and o["_date"] > end: return False
        return True
    rows = [o for o in rows_all if ok(o)]
    gb = spec.get("group_by")
    if not gb:
        return metric_value(m, rows)
    groups = {}
    for o in rows:
        key = o["_date"].strftime("%Y-%m") if gb == "month" else o[gb]
        groups.setdefault(key, []).append(o)
    vals = {k: metric_value(m, v) for k, v in groups.items()}
    rank = spec.get("rank")
    if rank:
        n = int(spec.get("n") or 1)
        order = sorted(vals.items(), key=lambda kv: (-kv[1] if rank == "top" else kv[1], kv[0]))
        names = [k for k, _ in order[:n]]
        return names[0] if n == 1 and names else names
    return vals

# ---------------------------------------------------------------- LLM planner (AI Pipe)
PROMPT = """Convert a question about an order ledger into a JSON query spec. Output ONLY JSON.
Fields (omit those not needed):
 metric: revenue_usd | refund_usd | refund_count | order_count | units | distinct_customers | avg_order_value_usd | list_products | list_regions
 status: paid | refunded | void | any   (omit for default: paid, or refunded for refund_* metrics)
 region, product, customer: exact filter value
 start, end: inclusive YYYY-MM-DD dates (business timezone IST)
 group_by: region | product | customer | month | status
 rank: top | bottom ; n: how many (use with group_by for 'which/most/least/best' questions)
Known regions: {regions}
Known products: {products}
Today (IST): {today}
"Refunds"/"refunded amount" => refund_usd; "how many refunds" => refund_count.
"Revenue"/"sales" => revenue_usd. "How many customers" => distinct_customers.
Question: {q}"""

def llm_spec(q):
    if not AIPIPE_TOKEN:
        return None
    rows = STATE["rows"]
    p = PROMPT.format(
        regions=sorted({o["region"] for o in rows}),
        products=sorted({o["product"] for o in rows}),
        today=datetime.now(IST).date().isoformat(), q=q)
    r = requests.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": p}],
        },
        timeout=9)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    return json.loads(txt)

# ---------------------------------------------------------------- rule-based fallback
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january","february","march","april","may","june","july","august","september","october","november","december"])}
for k, v in list(MONTHS.items()): MONTHS[k[:3]] = v

def month_range(y, m):
    s = date(y, m, 1)
    e = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return s, e

def rule_spec(q):
    ql = q.lower(); rows = STATE["rows"]; spec = {}
    if re.search(r"refund", ql):
        spec["metric"] = "refund_count" if re.search(r"how many|number|count", ql) else "refund_usd"
    elif re.search(r"how many customers|distinct customers|unique customers", ql):
        spec["metric"] = "distinct_customers"
    elif re.search(r"units|quantity", ql):
        spec["metric"] = "units"
    elif re.search(r"average|avg", ql):
        spec["metric"] = "avg_order_value_usd"
    elif re.search(r"how many orders|number of orders|order count", ql):
        spec["metric"] = "order_count"
    elif re.search(r"list.*products|which products|what products", ql):
        spec["metric"] = "list_products"
    else:
        spec["metric"] = "revenue_usd"
    for k in ("region", "product"):
        for v in sorted({o[k] for o in rows}, key=len, reverse=True):
            if re.search(r"\b" + re.escape(v.lower()) + r"\b", ql):
                spec[k] = v; break
    c = re.search(r"\bc\d{3,}\b", ql)
    if c: spec["customer"] = c.group(0).upper()
    y = re.search(r"\b(20\d\d)\b", ql)
    mm = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", ql)
    qq = re.search(r"\bq([1-4])\b", ql)
    if y and mm:
        s, e = month_range(int(y.group(1)), MONTHS[mm.group(1)])
        spec["start"], spec["end"] = str(s), str(e)
    elif y and qq:
        n = int(qq.group(1)); yr = int(y.group(1))
        spec["start"], spec["end"] = str(date(yr, 3 * n - 2, 1)), str(month_range(yr, 3 * n)[1])
    elif y:
        spec["start"], spec["end"] = f"{y.group(1)}-01-01", f"{y.group(1)}-12-31"
    top = re.search(r"\b(highest|most|top|best|largest|biggest)\b", ql)
    bot = re.search(r"\b(lowest|least|worst|smallest)\b", ql)
    if top or bot:
        for k in ("product", "region", "customer"):
            if re.search(k, ql) and k not in spec:
                spec["group_by"] = k; spec["rank"] = "top" if top else "bottom"; break
    return spec

# ---------------------------------------------------------------- HTTP API
@app.post("/")
@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    q = str(body.get("question", ""))
    if not STATE["rows"]:
        try:
            load()
        except Exception as e:
            print("on-demand load failed:", e)
    spec, ans = None, None
    try:
        spec = llm_spec(q)
        ans = execute(spec) if spec else None
    except Exception as e:
        print("llm path failed:", e)
        ans = None
    if ans is None:
        try:
            spec = rule_spec(q)
            ans = execute(spec)
        except Exception as e:
            print("rule path failed:", e)
            ans = None
    print(json.dumps({"q": q, "spec": spec, "answer": ans}, default=str))
    return {"answer": ans}

@app.get("/")
@app.get("/healthz")
async def health():
    return {"ok": True, "orders": len(STATE["rows"]), "raw_rows": STATE["raw"]}
