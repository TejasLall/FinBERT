# Presentation Notes: Automated Daily Newspaper Financial Sentiment & Market Impact Dashboard

These notes are the source material for the slide deck. Each `##` section maps to one or two slides. Every number below comes from an actual run of the app or its test scripts on 12–13 October 2026. Nothing is estimated unless marked as such.

---

## 1. The problem

- A daily financial newspaper carries dozens of stories, and reading all of them to judge the day's market mood takes time.
- **Goal:** upload a whole day's paper and get, in a few minutes:
  - the dominant market mood, for example "71% Bullish"
  - the sentiment of each story, with a confidence score
  - which stocks and indices are being talked about, and in what tone
  - the most bullish and most bearish headlines, filterable
- **Two technical problems** turned up along the way:
  1. **Multi-column layout.** A plain PDF text extractor reads straight across the columns, so neighbouring stories get interleaved.
  2. **Scrambled e-paper text.** The real test paper (Financial Express, Delhi, 12 Oct 2026, 42 pages) deliberately scrambles its text layer. pdfplumber, PyPDF2 and PDFium return `�` for 100% of characters, and PyMuPDF returns control codes such as `\x01\x02\x03`. Without OCR, the model would be scoring garbage.

## 2. Data sources

**There are no live news feeds or APIs.** The input is whatever file the user uploads:

| Input | How it's read |
|---|---|
| PDF e-paper with a readable text layer | PyMuPDF `page.get_text("blocks")` |
| PDF with a scrambled or scanned text layer | Pages rendered and read with RapidOCR |
| `.txt` | One headline per line, or articles separated by blank lines (auto-detected) |
| `.csv` | One article per row; the text column is auto-detected and can be changed |
| Single headline | Typed into the "Test a single headline" mode |

Test data used for this presentation:
- `Delhi-October-12-2026.pdf`: Financial Express e-paper, 42 pages, 1500 × 2384 pt pages, about 7–8 columns.
- `sample_news.txt`: 17 hand-written Indian and US market headlines, included in the project.
- A generated 2-page, 3-column test newspaper with known text, used to check the layout code exactly (test script only, not shipped).

## 3. Sentiment model

- **FinBERT (`ProsusAI/finbert`)** from Hugging Face: BERT-base, further pre-trained on financial text and fine-tuned for 3-class financial sentiment (positive, negative, neutral).
- **Inference details:**
  - **Batching:** articles are classified in mini-batches. The batch size is selectable (4–64) and defaults to 16.
  - **Length-sorted batching:** texts are sorted by length before batching, so short headlines aren't padded to the length of long articles.
  - **Tokenization:** WordPiece tokens with dynamic padding, truncation at **512 tokens**, and an attention mask.
  - **Gradients off:** inference runs under `torch.no_grad()` with the model in `eval()` mode.
  - **Probabilities:** a softmax over the logits. The predicted class is the arg-max, and its probability is the **confidence**.
  - **Labels:** read from `model.config.id2label`, never hard-coded.
- **Long articles** are split at sentence boundaries into chunks of at most **300 words**, so nothing is silently cut off by the 512-token limit.
- **Caching:** the model is loaded once per server (`@st.cache_resource`). Results are kept in session state, so moving a filter never re-runs the model.

## 4. How "market impact" is measured

**Important for the Q&A:** the project measures **sentiment about** stocks and indices. It does **not** measure price reactions. There is no price data, no event window and no returns calculation. "Impact" here means how strongly bullish or bearish the coverage is.

| Metric | Definition |
|---|---|
| **Impact score** (per article) | P(positive) − P(negative), from −1 (very bearish) to +1 (very bullish) |
| **Bullish %** (per day) | Positive ÷ (Positive + Negative). Neutral articles are excluded because factual reporting has no direction. |
| **Net score** (per day) | Mean impact score across all articles |
| **Ticker signal** | Mean impact score of every article that mentions the ticker. Above +0.15 is Bullish, below −0.15 Bearish, otherwise Neutral. |
| **Confidence** | Softmax probability of the predicted class |

**Tickers and indices tracked** (regex, no external API):
- **Watchlist:** 68 known symbols:
  - US large caps: AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META …
  - Indian large caps: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, TATAMOTORS …
  - Indices: **NIFTY, BANKNIFTY, SENSEX, NASDAQ, DOW, SPX (S&P 500), FTSE, NIKKEI, DAX**
- **Company names:** 72 names mapped to tickers, for example "Infosys" → INFY and "Bank Nifty" → BANKNIFTY.
- **Four matching rules:**
  1. cashtags such as `$TSLA`
  2. exchange prefixes such as `NSE:TCS`
  3. capitalised words, but only if they're on the watchlist, so CEO, GDP and RBI aren't counted as tickers
  4. company names
- Users can add extra tickers in the sidebar.
- **Time window:** one newspaper edition, meaning one day. There is no intraday or multi-day window.

## 5. Pipeline architecture

```
Upload (PDF / TXT / CSV / single headline)
   │
   ├─ PDF ─► Fast scan (PyMuPDF, 0.3 s for 42 pages): is each page's text layer readable?
   │          ├─ readable  ─► get_text("blocks") ─► sort blocks column by column
   │          │               ─► drop tiny blocks, running headers, captions, stock tables ─► chunks
   │          └─ scrambled ─► render page (110 dpi) ─► RapidOCR (English PP-OCRv5)
   │                          ─► headline detection by text height ─► each headline takes
   │                             the text beneath it, within its width ─► articles
   ├─ TXT ─► clean + split (one item per line, or paragraph blocks)
   └─ CSV ─► one row = one article
   │
   ▼
Chunk to ≤ 300 words ─► FinBERT batched inference (progress bar) ─► softmax probabilities
   │
   ▼
Regex ticker extraction ─► aggregation (mood, distribution, per-ticker signal, top headlines)
   │
   ▼
Streamlit dashboard (Plotly charts, filterable tables, CSV export, layout overlay)
```

**Tech stack** (versions installed and tested):
- Python 3.13, Streamlit 1.65
- transformers 5.19, torch 2.14 (CPU)
- PyMuPDF 1.28, RapidOCR 3.10 + onnxruntime 1.31
- pdfplumber 0.11 (for the old plain-text mode)
- pandas 3.0, Plotly 7.1

## 6. Key results

### 6a. The real newspaper (Financial Express, Delhi, 12 Oct 2026)

| Measurement | Result |
|---|---|
| Pages with a scrambled text layer | **42 / 42**, detected in 0.3 s |
| OCR time per page (CPU, 110 dpi) | **17.9 s** (p1), **23.0 s** (p3), **26.2 s** (p10) |
| End-to-end run, pages 1–2 | OCR 38 s → **30 chunks** from 104 text regions, **23 headlines** |
| FinBERT on those 30 chunks | **6 s** |
| Dashboard for pages 1–2 | **71% Bullish**, 10 positive / 4 negative, **81% average confidence**, 7 tickers |
| Pages 1, 3 and 10 combined | 127 text regions, 27 headlines; layout analysis takes 0.06 s once OCR is cached |

Stories correctly separated with their full titles (pages 3 and 10):
- "Rice exports may take a hit next year"
- "NTPC seeking role beyond generation"
- "Registrations of EVs jump 84%; UP leads among states"
- "Barricading Young India"
- "Jio changed telecom, Starlink faces the test"
- "The employee who does just enough"

### 6b. Engineering results worth a slide

- **OCR tuning on the densest page (p10):**
  - Time per page fell from 102 s to **31 s**, by switching to the English PP-OCRv5 recognizer with 32-line batches.
  - On a test crop, words wrongly run together fell from **17 to 7** ("offthenational capitalin" → "off the national capital in").
- **Headline threshold calibrated from real measurements:** pull quotes measure **≤ 1.23×** body text height, and the smallest real headlines **≥ 1.31×**. So the cut-off is **1.28×**.
- **Generated 3-column test paper** (4 known stories): both the PyMuPDF block path and the OCR path recover every story with **no text mixed between stories**. Headers, footers, captions and the stock table are all removed.

### 6c. Sample headlines (`sample_news.txt`, 17 headlines)

- Mood: **53% Bearish** (7 positive / 8 negative / 2 neutral), net score **−0.09**
- **93% average confidence**, 17 distinct tickers detected
- Single-headline check: "Tesla shares plunge 12% after weaker-than-expected quarterly deliveries" → **Negative**

**Caveat for the slides:** these are demo runs on small samples. The sentiment labels have **not** been compared with human-labelled ground truth, so there is no accuracy figure for this data.

## 7. Limitations

- **No price data:** "market impact" is the tone of the coverage, not a measured stock-price reaction (see section 4).
- **No accuracy evaluation** against human labels on this newspaper.
- **OCR is slow on CPU:** about 18–26 s per page, so a full 42-page paper takes roughly 15–20 minutes (an estimate). Pages are cached once read, and the app reads pages 1–8 by default.
- **Layout heuristics:**
  - Large-type pull quotes, decks and bylines are handled by rules tuned on this paper; other papers may need tuning.
  - Stories that continue on another page ("Continued on page 7") end up as separate pieces.
  - Long stories become several "part 1/2" chunks.
- **Noise reaches the model:** bank legal notices and ads also come through as chunks. FinBERT mostly rates them neutral, but they still dilute the counts.
- **Ticker matching:** only symbols on the watchlist are found unless the user adds more. Company names are matched case-insensitively, so ambiguous names can cause false matches.
- **Article-level sentiment:** one label per article or chunk, so mixed-tone articles are averaged rather than split by entity.
- **Single-language:** English only.

## 8. Next steps

1. **Measure real market impact:** pull prices (for example from an exchange data API or Yahoo Finance) and run an event study. Compare each ticker's sentiment with its return over the next 1 and 5 trading days, against NIFTY/SENSEX as the benchmark.
2. **Evaluate accuracy:** hand-label 100–200 articles and report accuracy and F1 for FinBERT on Indian financial news.
3. **Entity-level sentiment:** score the sentences that mention each ticker, instead of the whole article.
4. **Filter ads and legal notices** before classification, using a small classifier or keyword rules.
5. **Faster OCR:** use a GPU, or OCR only the pages the user picks.
6. **History:** store each day's results to plot a mood trend over weeks.

## 9. Files, charts and outputs

### Project files (`C:\Users\Tejas Lall\financial-sentiment-dashboard\`)

| File | What it is |
|---|---|
| `app.py` | The whole application, about 2,240 lines. Look for the "VIVA NOTE" comments. |
| `requirements.txt` | Python dependencies |
| `sample_news.txt` | 17 demo headlines |
| `.streamlit/config.toml` | Dark "fintech" theme (Inter font, green accent) |
| `presentation_assets/layout_overlay_page1.png` | FE page 1, one numbered box per detected article |
| `presentation_assets/layout_overlay_page3.png` | FE page 3 (rice exports / NTPC / EV registrations) |
| `presentation_assets/layout_overlay_page10.png` | FE page 10 (Opinion page) |

The three overlay images are made with the final code. They are the best visual for the column-detection slide.

### Charts in the app

These are drawn live in the dashboard; **none are saved to disk**:
- **Market mood card:** Bullish %, net score and a proportional positive/negative/neutral bar
- **Sentiment mix:** donut chart (Plotly)
- **Sentiment by ticker:** stacked horizontal bar chart (Plotly), with a top-N slider
- **Ticker leaderboard:** table of net score and signal per ticker
- **Top impact headlines:** filterable tables of the most positive and most negative articles
- **Class probabilities:** bar chart in single-headline mode
- **Show detected layout:** toggle for PDFs that produces the overlay images above

### Exports

- **"Download results as CSV"** in the *All articles* tab saves `finbert_daily_sentiment.csv`, with one row per article: text, sentiment, confidence, impact score, tickers and page.

### Screenshots to take for the deck

No dashboard screenshot is currently saved. To capture them:
1. Run `streamlit run app.py`.
2. Upload `sample_news.txt` and click **Run FinBERT**. Screenshot the mood cards, the donut and ticker chart row, and the headlines table.
3. Upload the FE PDF, read pages 1–2, and screenshot the dashboard ("71% Bullish") and the **Show detected layout** view.
4. Switch to **Test a single headline** and screenshot the probability chart for the Tesla headline.
