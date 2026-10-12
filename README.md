Financial Sentiment & Market Impact DashboardAn end-to-end NLP and Computer Vision pipeline that extracts, parses, and scores financial news from full-page e-papers, text files, or raw text. Built with Streamlit, FinBERT, PyMuPDF, and RapidOCR, this tool processes complex multi-column newspaper layouts and scrambled text layers to measure aggregate market mood and entity-level sentiment.FeaturesMulti-Source Ingestion: Supports PDF e-papers (digital and scrambled/scanned), .txt headline dumps, .csv article datasets, and manual single-headline testing.Robust e-Paper Layout Parsing:Automatically detects scrambled or corrupted PDF text layers using PyMuPDF fast scanning (~0.3s for 42 pages).Uses RapidOCR (PP-OCRv5) with custom batching for scanned/scrambled pages.Smart headline detection using font height ratios ($\ge 1.28\times$ body text height) and column-aware bounding box association.Ignores non-content artifacts like running headers, footers, stock price tables, and captions.Optimized FinBERT Inference:Uses ProsusAI/finbert fine-tuned for 3-class financial sentiment (positive, negative, neutral).Implements sentence-aware chunking ($\le 300$ words) to prevent truncation within FinBERT's 512-token limit.Length-sorted mini-batching (default batch size: 16) with torch.no_grad() evaluation on CPU.Cached model loading via @st.cache_resource.Ticker & Entity Extraction:Extracted using pattern-matching across 68 watchlist symbols (US/Indian equities, major indices like NIFTY, BANKNIFTY, SENSEX, NASDAQ) and 72 company-to-ticker maps.Supports cashtags ($TSLA), exchange prefixes (NSE:TCS), capitalized watchlist symbols, and full company name mappings.Interactive Analytics:Real-time market mood indicator (Bullish % excluding neutral coverage, Net Impact Score).Interactive Plotly visualizations: sentiment distribution donuts, per-ticker sentiment breakdowns, and top impact headline tables.Bounding box overlay toggle to inspect detected newspaper regions.Single-click CSV export (finbert_daily_sentiment.csv).Pipeline Architecture                       ┌──────────────────────────────┐
                       │   Input (PDF / TXT / CSV)    │
                       └──────────────┬───────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   [ Readable PDF / TXT / CSV ]                   [ Scrambled / Scanned PDF ]
              │                                               │
   PyMuPDF block extraction                         RapidOCR (PP-OCRv5)
   Column sorting & cleaning                        Headline height detection
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Sentence-Aware Chunking    │
                      │       (≤ 300 words/chunk)     │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   FinBERT Batched Inference   │
                      │  (ProsusAI/finbert + Softmax) │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Regex Ticker Matcher &     │
                      │    Market Signal Generator    │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │      Streamlit Dashboard      │
                      │   (Plotly, Overlays, CSV)     │
                      └───────────────────────────────┘
Methodology & MetricsNote: This dashboard measures sentiment intensity in financial news coverage, not stock price returns or market reaction window calculations.MetricDefinition / CalculationImpact Score$P(\text{positive}) - P(\text{negative})$, bounded between $-1.0$ (Strong Bearish) and $+1.0$ (Strong Bullish).Bullish %$\frac{\text{Positive Count}}{\text{Positive Count} + \text{Negative Count}} \times 100$ (Neutral articles excluded)Net ScoreArithmetic mean of Impact Scores across all processed articles/chunks.Ticker SignalMean impact score for articles referencing a specific ticker:• Bullish: $> +0.15$• Bearish: $< -0.15$• Neutral: OtherwiseConfidenceSoftmax probability of the predicted sentiment class.Tech StackLanguage & UI: Python 3.13, Streamlit 1.65NLP & Deep Learning: PyTorch 2.14 (CPU), Hugging Face Transformers 5.19 (ProsusAI/finbert)Document Processing & OCR: PyMuPDF 1.28, RapidOCR 3.10, ONNX Runtime 1.31, pdfplumber 0.11Data Analytics & Visualization: Pandas 3.0, Plotly 7.1Getting StartedPrerequisitesEnsure you have Python 3.10+ installed. A virtual environment is recommended.Bash# Clone the repository
git clone https://github.com/TejasLall/FinBERT.git
cd FinBERT

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Running the ApplicationBashstreamlit run app.py
Open your browser at http://localhost:8501.Usage GuideFull Newspaper Ingestion (PDF):Upload a financial daily e-paper (e.g., Financial Express, Economic Times).The app auto-detects if the text layer is scrambled. If scrambled, OCR runs per page.Toggle "Show detected layout" in the sidebar to view bounding box overlays identifying headlines and column groupings.Text / CSV Bulk Processing:Upload a .txt file containing headlines (one per line) or articles separated by blank lines.Upload a .csv file and select the target article/headline text column.Single Headline Testing:Navigate to the "Test a single headline" tab to paste any text and visualize class probabilities instantly.Exporting Data:Export processed results directly from the All Articles tab to finbert_daily_sentiment.csv.Benchmarks & EvaluationTested on Financial Express (Delhi Edition, 42 pages):Fast Text Scan: Identified scrambled text layers across 42 pages in 0.3 seconds.OCR Execution (CPU): ~18–26 seconds per dense page using PP-OCRv5 with 32-line mini-batches.Layout Precision: Headline threshold calibrated at $1.28\times$ body text height, cleanly separating pull quotes ($\le 1.23\times$) from actual headlines ($\ge 1.31\times$).Inference Speed: 30 document chunks evaluated through FinBERT in ~6 seconds on CPU.Project Structurefinancial-sentiment-dashboard/
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
├── presentation_assets/
│   ├── layout_overlay_page1.png  # Sample layout extraction overlay
│   ├── layout_overlay_page3.png
│   └── layout_overlay_page10.png
├── app.py                        # Core Streamlit application & processing pipeline
├── requirements.txt              # Project dependencies
├── sample_news.txt               # Demo dataset (17 test headlines)
└── README.md
Limitations & Future EnhancementsCurrent Limitations:CPU-bound OCR takes 15–20 minutes for a full 42-page paper (mitigated by page caching and processing pages 1–8 by default).Legal notices, bank ads, and disclosures are currently parsed alongside news stories.Multi-page article jump continuations ("Continued on page 7") are evaluated as separate segments.Future Roadmap:Market Event Studies: Integrate Yahoo Finance / NSE APIs to correlate sentiment scores with 1-day and 5-day post-publication stock returns.Entity-Level Sentiment: Score individual sentences mentioning a target entity rather than whole-article aggregation.Ad & Disclosure Filtering: Train a lightweight classifier to scrub bank notices and non-editorial advertisements before sentiment scoring.GPU Acceleration: Add CUDA support for OCR and FinBERT inference pipelines.
