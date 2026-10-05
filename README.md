# ETF & Portfolio Analysis Tool

Questa è un'applicazione web completa, costruita con Streamlit, che offre una suite di strumenti potenti per l'analisi approfondita di ETF e portafogli di investimento. È stata progettata per essere intuitiva e potente, adatta sia per investitori neofiti che per esperti.

## Funzionalità

-   **Portfolio Tracker**: Monitora e analizza i tuoi portafogli di investimento in modo dettagliato, con:
    -   Creazione e gestione di portafogli multipli
    -   Registrazione di transazioni di acquisto e vendita
    -   Visualizzazione di un riepilogo con le metriche chiave
    -   Analisi delle posizioni aperte con P&L non realizzato e segnali di trend
    -   Consultazione dello storico delle transazioni e del cassetto fiscale con P&L realizzato
-   **Portfolio Backtester**: Simula e confronta le performance di portafogli personalizzati nel tempo, con:
    -   Scelta tra strategie di investimento "Lump Sum (PIC)" e "PAC"
    -   Definizione manuale o caricamento da portafogli esistenti
    -   Impostazione della frequenza di ribilanciamento
    -   Confronto con portafogli modello (es. 60/40, All-Weather)
-   **Screener Tattico PAC**: Scopri ETP (Exchange Traded Products) con potenziale di breakout quotati sulle principali borse europee, basato su un approccio multi-fattore.
-   **Trading & Backtest**: Suite di **5 strategie algoritmiche** con l'obiettivo di battere il S&P500, con scanner di segnali, backtest singolo e confronto multi-strategia:
    - **🧠 AI Enhanced Momentum** (strategia originale): filtro macro, ranking AI Score, stop ATR e take profit parziale
    - **🌊 Trend Fusion**: fusione dei tre algoritmi migliori del confronto storico — entry in confluenza (breakout Donchian 55 + Value Area High + volume ≥ 1.5x + ADX/RS), TP parziale con lock a breakeven, uscita Donchian LL20 che lascia correre il trend
    - **🐻 Bear Market Regime Switcher**: compra i forti in bull, cerca inversi/difensivi in bear, cash nei regimi di crisi
    - **📊 Volume Profile**: breakout dal Value Area con conferma di volume (POC/VAH/VAL rolling)
    - **🐢 Turtle Breakout**: breakout Donchian 55/20 in stile Turtle Trading
    - **Dati fake**: 5 scenari di mercato sintetici (bull, bear, crash, laterale, misto) per simulare senza connessione a Yahoo Finance, con universo di archetipi (tech, oro, obbligazioni, ETF inverso, ...)
    - Report con KPI commissioni (totale, operazioni/anno, impatto sul capitale) e confronto con S&P500 Buy & Hold
-   **Portafoglio Live**: Gestione operativa di un portafoglio algoritmico con stop loss / take profit trailing, sizing dinamico e regime di mercato.

## Struttura del Progetto

```
portfolio_tracker/
├── app.py                          # Entrypoint per l'avvio dell'applicazione Streamlit
├── pyproject.toml                  # Dipendenze e configurazione del progetto (uv)
├── README.md                       # Questo file
└── etf_metrics/
    ├── ui/
    │   ├── main.py                 # Navigazione principale e stato della sessione
    │   ├── portfolio_tracker.py    # UI per il tracciamento di portafogli
    │   ├── portfolio_backtester.py # UI per il backtesting di portafogli
    │   ├── pac_screener.py         # UI per lo screener tattico PAC
    │   ├── trading.py              # UI per scanner e backtest di trading
    │   └── algo_live.py            # UI per il portafoglio algoritmico live
    ├── core/
    │   ├── portfolio_tracker.py    # Logica per il tracciamento di portafogli
    │   ├── portfolio_backtester.py # Logica per il backtesting di portafogli
    │   ├── pac_screener.py         # Logica per lo screener tattico PAC
    │   ├── trading.py              # Logica per i segnali di trading
    │   ├── automated_backtest.py   # Logica per il backtest algoritmico
    │   ├── etf_search_engine.py    # Logica di ricerca e filtraggio degli ETF
    │   ├── data_manager.py         # Cache locale (DuckDB/SQLite) delle serie
    │   │                           # storiche + blacklist persistente dei ticker
    │   │                           # senza dati (stessa API per entrambi i backend)
    │   └── metrics.py              # Funzioni per le metriche di performance e rischio
    ├── clients/
    │   ├── base_client.py          # Classe base per i client esterni (retry, errori)
    │   └── yahoo_client.py         # Client Yahoo Finance: download bulk in batch,
    │   │                           # cache negativa dei ticker "no data found",
    │   │                           # recupero parallelo di nomi/ISIN
    └── shared/
        ├── config.py               # Costanti, configurazioni e portafogli modello
        └── utils.py                # Funzioni di utilità generiche
├── scripts/
    ├── simulate_strategies.py      # Simulatore CLI multi-strategia su dati fake
    ├── analyze_db.py               # Analisi finanziaria offline del DB dei prezzi
    ├── test_data_hub.py            # Test funzionali Gestione Dati + UI (76 test)
    ├── test_dual_backend.py        # Parità SQLite/DuckDB del data manager
    ├── test_determinism.py         # Il backtest ottimizzato dà risultati identici
    ├── test_perf_compare.py        # Benchmark velocità motore di backtest
    └── test_bug_repro.py           # Riproduzione del bug "metriche a 0" (fixed)
```

## Requisiti

Per l'elenco completo delle dipendenze, si veda il file `pyproject.toml`. È richiesto **Python ≥ 3.9**.
La gestione delle dipendenze avviene tramite [uv](https://docs.astral.sh/uv/).

## Setup Rapido

1.  **Installa uv** (se non lo hai già):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sincronizza le dipendenze** (crea automaticamente l'ambiente virtuale `.venv` e installa tutto):

    ```bash
    uv sync
    ```

## Avvio

Per avviare l'applicazione, esegui il seguente comando dalla root del progetto:

```bash
uv run streamlit run app.py
```

`uv run` usa automaticamente l'ambiente virtuale del progetto, senza doverlo attivare manualmente.

## Simulazioni senza connessione (dati fake)

Se Yahoo Finance non è raggiungibile (o per stress-testare le strategie) puoi usare il simulatore CLI:

```bash
# Scenario misto, 4 anni, tutte le strategie
python3 scripts/simulate_strategies.py

# Stress test ribassista con più seed (robustezza)
python3 scripts/simulate_strategies.py --scenario bear_market --seeds 1 7 42

# Solo due strategie, broker costoso (5€/operazione)
python3 scripts/simulate_strategies.py --strategies turtle_breakout trend_fusion --commission 5
```

Stessa cosa dall'interfaccia: **Trading → Automated Backtest / Confronto Strategie → Fonte Dati: Dati simulati (Fake)**.

> ⚠️ Nota sui dati fake: servono a validare la *logica* delle strategie nei vari regimi di mercato
> (bull, bear, crash, laterali), non a prevedere i rendimenti reali. Un algoritmo che vince solo
> su un determinato seed/scenario è probabilmente in overfitting: verifica sempre su più seed
> e su più scenari prima di fidarti.

## 📥 Gestione Dati (download una volta, poi solo delta)

La modalità **📥 Gestione Dati** dell'app scarica lo storico dei ticker che inserisci
nel DB SQLite locale e, ai giri successivi, aggiorna **solo le righe mancanti**:

- **Ticker nuovo** → download completo dello storico scelto (5y/10y/2y/max)
- **Ticker già nel DB** → download delta dall'ultima data salvata (ultima giornata
  inclusa, per correggere prezzi parziali) e upsert incrementale
- I ticker con la stessa ultima data vengono **raggruppati in un solo batch**
  `yf.download` (download parallelo, minimizza le richieste)
- Report con righe nuove per ticker, coverage date, ticker saltati (blacklist /
  rate limit) e tempo impiegato
- **🌍 Aggiorna tutto il DB**: delta-update dell'intero universo in un click,
  con skip opzionale dei già sincronizzati oggi
- Ispezione del contenuto del DB (ticker, nome, righe, copertura, ultimo
  aggiornamento) e della cache negativa, con reset

Il DB dei prezzi è **DuckDB** (`market_data.duckdb`, creato automaticamente al
primo download): un motore **colonnare embedded** (nessun server, come SQLite)
ma 10-50x più veloce sulle letture analitiche — decisivo con universi grandi
(es. tutto Xetra, ~4 milioni di righe). Un eventuale vecchio `market_data.db`
SQLite resta al suo posto invariato e resta leggibile (il backend è scelto dal
percorso: `.duckdb` = DuckDB, altrimenti SQLite). Il percorso è configurabile
con la variabile d'ambiente `ETF_METRICS_DB`.

## Ottimizzazione download dati (universi grandi, es. tutto Xetra)

Il client Yahoo è ottimizzato per scaricare migliaia di ticker:

- **`get_series_bulk(tickers)`**: download in **batch da 200 ticker con 16 thread**
  (configurabile in `etf_metrics/shared/config.py`). Un batch fallito non blocca
  l'universo, e un `progress_callback` mostra l'avanzamento in UI.
- **Cache negativa dei "no data found"**: i ticker che non hanno dati vengono salvati
  nella tabella `failed_tickers` del DB e **non vengono mai più riscaricati**
  (policy `FAILED_TICKER_RETRY_DAYS`, di default permanente). Un risultato vuoto finisce
  in blacklist **solo se Yahoo risulta raggiungibile** (health-check), così un
  problema di rete non avvelena la blacklist. Se un ticker prima fallito produce
  dati, viene riabilitato automaticamente.
- **Gestione del rate limit Yahoo (HTTP 429)**: quando Yahoo risponde "Too Many
  Requests" i download vengono **sospesi automaticamente per 10 minuti**
  (`YAHOO_RATE_LIMIT_COOLDOWN`) invece di martellare il server — il 429 viene
  rilevato sia come eccezione sia dai log interni di yfinance. Durante il cooldown
  nessun ticker finisce in blacklist e la UI lo segnala chiaramente con l'attesa
  residua. Un bulk interrotto a metà restituisce comunque i dati già scaricati.
  Tra un batch e l'altro viene inoltre applicata una piccola pausa
  (`BULK_BATCH_PAUSE`, 1s di default) per ridurre la probabilità di far scattare il limite.
- **Nomi/ISIN in parallelo**: `get_names_bulk` / prefetch del motore di ricerca ETF
  usano un thread pool (16 worker) invece di migliaia di chiamate sequenziali.
- **Query SQL a chunk**: `load_data` / `get_tickers_needing_update` gestiscono
  5000+ ticker senza problemi di limite parametri SQLite.
- **Primo download automatico**: nella modalità "Dati reali" del Trading UI il
  checkbox "📥 Scarica da Yahoo i ticker mancanti" (attivo di default) popola il DB
  al primo avvio; i giri successivi usano solo la cache locale. I caricamenti
  falliti non vengono memorizzati in cache: il retry funziona sempre.

Dalla UI dello Screener ETF (sidebar) è possibile ispezionare e resettare la blacklist
dei ticker senza dati. Benchmark con latenza simulata: **~20x più veloce** del
download sequenziale, a cui si aggiunge l'azzeramento dei re-download dei ticker morti.

> 💡 **Suggerimento (test / multi-istanza):** il percorso del DB dei prezzi è
> sovrascrivibile con la variabile d'ambiente `ETF_METRICS_DB=/percorso/test.db`
> per eseguire simulazioni e test senza toccare il database di produzione.

## 🏁 Benchmark di confronto: S&P500

Il benchmark di riferimento del Trading (curva Buy & Hold, filtro macro di
regime, RS dei titoli) è **SPY**, l'ETF sull'S&P500. `prepare_market_data` lo
**aggiunge automaticamente** all'universo scaricato: la prima volta che esegui
un backtest con l'aggiornamento da Yahoo attivo, SPY viene scaricato e salvato
nel DB — da lì in poi resta disponibile anche offline.

Se SPY non è nel DB (es. run offline con DB appena creato) l'app usa come
riparo un **indice equal-weight dell'universo** (etichetta "Universo (proxy)"
nelle curve) e la pagina Trading lo segnala con un pulsante per scaricare SPY
subito. Il proxy è un ripiego: per confrontare le performance con l'S&P500
tienilo nel DB.

## ⚡ Performance del motore di backtest

Il motore di `automated_backtest.py` è ottimizzato per universi grandi
(tutto Xetra, ~4M di righe) e multi-core:

- **Regimi di mercato precalcolati** in un'unica passata vettoriale
  (precedenza BEAR > VOLATILE > DANGER > BULL), invece di rivalutarli ogni giorno
- **Volume profile rolling vettoriale** (`sliding_window_view` + `bincount`,
  ~5x più veloce del loop originale)
- **Pannello prezzi numpy** per il motore day-by-day (uscite/entrate/NAV:
  ~5x più veloce delle lookup `.loc`)
- **Cache con fingerprint** dei DataFrame preprocessati (indicatori, SMA, ATR:
  calcolati una volta sola per strategia e riusati)
- **Confronto strategie in parallelo** con `ProcessPoolExecutor` (fork): le 7
  strategie girano su core diversi e condividono i dati senza serializzazione.
  Il numero di worker è limitato a `min(strategie, CPU, 4)` e regolabile con
  `ETF_METRICS_BACKTEST_WORKERS`; il budget della cache con
  `ETF_METRICS_PREPARED_CACHE_ROWS` (default 6M righe)

Risultato su universo sintetico da 1.300 ticker / 4M di righe (i9, 10 anni di
simulazione): **tutte le strategie in ~67s** invece di ~10 minuti; letture dal DB
**~4x più veloci** con DuckDB rispetto a SQLite. Tutti i risultati numerici
sono **identici** al motore originale (verificato da `test_determinism.py`).

### Benchmark e analisi offline

```bash
# Analisi finanziaria del DB (performance, correlazioni, regime attuale)
python3 scripts/analyze_db.py

# Benchmark del motore su 4M di righe sintetiche (nessuna rete)
python3 scripts/bench_xetra_scale.py

# Parità dei due backend del data manager
python3 scripts/test_dual_backend.py
```
