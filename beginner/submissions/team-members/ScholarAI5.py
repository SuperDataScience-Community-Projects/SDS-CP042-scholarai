# app.py — ScholarAI (Streamlit + SerpAPI, single file)

# Requirements (put these in requirements.txt and pip install them first):
# streamlit==1.39.0
# python-dotenv==1.0.1
# openai==1.52.2
# serpapi==0.1.5  # PyPI name is google-search-results
# pydantic==2.9.2
# tiktoken==0.7.0

import os, json, re
from typing import List, Dict
from urllib.parse import urlparse
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch
from openai import OpenAI
from pydantic import BaseModel, Field, HttpUrl

# ------------------------ Helpers & Schemas ------------------------
class CuratedSource(BaseModel):
    title: str
    url: HttpUrl
    snippet: str
    score: float = 0.0

class ResearchBundle(BaseModel):
    query: str
    sources: List[CuratedSource]

class KeyFinding(BaseModel):
    text: str
    citation_urls: List[HttpUrl] = Field(default_factory=list)

class Report(BaseModel):
    tldr: str
    key_findings: List[KeyFinding]
    conflicts_and_caveats: List[str]
    top_links: List[str]

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc.lower().replace("www.", "")
    except Exception:
        return u

def _score_item(query: str, title: str, snippet: str, url: str) -> float:
    q_tokens = set(normalize_text(query).split())
    text = normalize_text(f"{title} {snippet}")
    hits = sum(1 for t in q_tokens if t in text)
    bonus = 0.5 if any(suf in _domain(url) for suf in [".edu", ".gov"]) else 0.0
    return hits + bonus

# ------------------------ Tools ------------------------
def web_search(query: str, k: int = 10) -> List[Dict]:
    """SerpAPI: returns [{title, url, snippet}]"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY not set. Add it in .env or sidebar.")
    params = {
        "engine": "google",
        "q": query,
        "num": min(max(k, 1), 20),
        "api_key": api_key,
        "hl": "en",
    }
    results = GoogleSearch(params).get_dict()
    organic = results.get("organic_results", []) or []
    out = []
    for it in organic:
        title = it.get("title") or ""
        link = it.get("link") or it.get("url") or ""
        snippet = it.get("snippet") or ""
        if title and link:
            out.append({"title": title, "url": link, "snippet": snippet})
    return out[:k]

def curate_sources(query: str, k: int = 10, top_n: int = 8) -> ResearchBundle:
    raw = web_search(query, k)
    seen = set()
    curated = []
    for r in raw:
        s = _score_item(query, r["title"], r["snippet"], r["url"])
        d = _domain(r["url"])
        if d not in seen:
            seen.add(d)
            curated.append(CuratedSource(title=r["title"], url=r["url"], snippet=r["snippet"], score=s))
    curated.sort(key=lambda x: x.score, reverse=True)
    return ResearchBundle(query=query, sources=curated[:top_n])

SYSTEM_PROMPT = """You are a precise research synthesizer.
Return STRICT JSON of this schema:

{
  "tldr": str (<=120 words),
  "key_findings": [{"text": str, "citation_urls": [str]}],
  "conflicts_and_caveats": [str],
  "top_links": [str]
}

Rules:
- Use only given sources.
- Cite at least 3 if available.
- Neutral, concise, fact-based.
- Output JSON only (no prose, no code fences).
"""

def _sources_block(bundle: ResearchBundle) -> str:
    return "\n\n".join(
        f"[{i+1}] {s.title}\nURL: {s.url}\nSnippet: {s.snippet}" for i, s in enumerate(bundle.sources)
    )

def synthesize(bundle: ResearchBundle, model: str = "gpt-4o-mini") -> Report:
    client = OpenAI()
    user_prompt = f"""Query: {bundle.query}

Sources:
{_sources_block(bundle)}

Task: Produce the JSON report."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    if "```" in raw:  # in case model wraps JSON
        raw = raw.split("```")[1]
        if raw.strip().startswith("json"):
            raw = raw.split("\n", 1)[1]
    data = json.loads(raw)

    kf_objs = [KeyFinding(text=k["text"], citation_urls=k.get("citation_urls", []))
               for k in data.get("key_findings", [])]
    return Report(
        tldr=data.get("tldr", ""),
        key_findings=kf_objs,
        conflicts_and_caveats=data.get("conflicts_and_caveats", []),
        top_links=data.get("top_links", []),
    )

def to_json(report: Report) -> str:
    return report.model_dump_json(indent=2)

def to_markdown(report: Report) -> str:
    lines = ["# TL;DR", report.tldr.strip(), "\n# Key Findings"]
    for i, kf in enumerate(report.key_findings, 1):
        cites = ", ".join(f"[{j+1}]({u})" for j, u in enumerate(kf.citation_urls))
        lines.append(f"- **{i}.** {kf.text.strip()} " + (f"({cites})" if cites else ""))
    lines.append("\n# Conflicts & Caveats")
    if report.conflicts_and_caveats:
        lines += [f"- {c}" for c in report.conflicts_and_caveats]
    else:
        lines.append("- None noted.")
    lines.append("\n# Top 5 Links")
    lines += [f"{i+1}. {u}" for i, u in enumerate(report.top_links[:5])]
    return "\n".join(lines) + "\n"

# ------------------------ Streamlit UI ------------------------
load_dotenv()  # load .env if present
st.set_page_config(page_title="ScholarAI (SerpAPI)", page_icon="🔎", layout="wide")
st.title("🔎 ScholarAI — SerpAPI Research & Synthesis")

with st.sidebar:
    st.header("Settings")
    # Allow setting keys from the sidebar (handy for local demos)
    openai_key_in = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    serp_key_in = st.text_input("SERPAPI_API_KEY", value=os.getenv("SERPAPI_API_KEY", ""), type="password")
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    k = st.slider("Search results (k)", 5, 20, 12)
    top_n = st.slider("Top sources used", 3, 10, 8)
    st.caption("Keys entered here override environment variables for this session.")
    if openai_key_in:
        os.environ["OPENAI_API_KEY"] = openai_key_in
    if serp_key_in:
        os.environ["SERPAPI_API_KEY"] = serp_key_in

query = st.text_input("Your research question", value=os.getenv("QUERY", "What is retrieval-augmented generation (RAG)?"))
run = st.button("Run research")

placeholder = st.empty()

if run:
    try:
        with st.spinner("Searching with SerpAPI…"):
            bundle = curate_sources(query, k=k, top_n=top_n)

        st.subheader("Curated Sources")
        for i, s in enumerate(bundle.sources, 1):
            with st.expander(f"[{i}] {s.title}"):
                st.write(f"**URL:** {s.url}")
                st.write(s.snippet)
                st.write(f"Relevance score: `{s.score:.2f}`")

        with st.spinner("Synthesizing with OpenAI…"):
            report = synthesize(bundle, model=model)

        st.subheader("Report")
        md = to_markdown(report)
        st.markdown(md)

        # Downloads
        out_dir = Path("out"); out_dir.mkdir(exist_ok=True, parents=True)
        json_text = to_json(report)
        (out_dir / "report.json").write_text(json_text, encoding="utf-8")
        (out_dir / "report.md").write_text(md, encoding="utf-8")

        st.download_button("⬇️ Download report.json", data=json_text, file_name="report.json", mime="application/json")
        st.download_button("⬇️ Download report.md", data=md, file_name="report.md", mime="text/markdown")

        st.success("Done!")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("Enter a query and press **Run research**.")
