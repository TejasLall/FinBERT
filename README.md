Financial Sentiment & Market Impact DashboardAn end-to-end NLP and Computer Vision pipeline that extracts, parses, and scores financial news from full-page e-papers, text files, or raw text. 
Built with Streamlit, FinBERT, PyMuPDF, and RapidOCR, this tool processes complex multi-column newspaper layouts and scrambled text layers to measure aggregate market mood and entity-level sentiment.FeaturesMulti-Source Ingestion: Supports PDF e-papers (digital and scrambled/scanned), .txt headline dumps, .csv article datasets, and manual single-headline testing.Robust e-Paper Layout Parsing
Automatically detects scrambled or corrupted PDF text layers using PyMuPDF fast scanning (~0.3s for 42 pages).
Uses RapidOCR (PP-OCRv5) with custom batching for scanned/scrambled pages.Smart headline detection using font height ratios ($\ge 1.28\times$ body text height) and column-aware bounding box association.
Ignores non-content artifacts like running headers, footers, stock price tables, and captions.Optimized FinBERT Inference:Uses ProsusAI/finbert fine-tuned for 3-class financial sentiment (positive, negative, neutral).
Implements sentence-aware chunking ($\le 300$ words) to prevent truncation within FinBERT's 512-token limit.Length-sorted mini-batching (default batch size: 16) with torch.no_grad() evaluation on CPU.Cached model loading via @st.cache_resource.Ticker & Entity Extraction:
Extracted using pattern-matching across 68 watchlist symbols (US/Indian equities, major indices like NIFTY, BANKNIFTY, SENSEX, NASDAQ) and 72 company-to-ticker maps.Supports cashtags ($TSLA), exchange prefixes (NSE:TCS), capitalized watchlist symbols, and full company name mappings
Interactive Analytics:Real-time market mood indicator (Bullish % excluding neutral coverage, Net Impact Score).
Interactive Plotly visualizations: sentiment distribution donuts, per-ticker sentiment breakdowns, and top impact headline tables.
Bounding box overlay toggle to inspect detected newspaper regions.
Single-click CSV export (finbert_daily_sentiment.csv).
