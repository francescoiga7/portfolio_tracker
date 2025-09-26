# ETF & Portfolio Analysis Tool

Questa è un'applicazione web completa, costruita con Streamlit, che offre una suite di strumenti potenti per l'analisi approfondita di ETF e portafogli di investimento. È stata progettata per essere intuitiva e potente, adatta sia per investitori neofiti che per esperti.

## Funzionalità

-   **Analisi Singolo ETF**: Calcola un'ampia gamma di metriche di performance e di rischio per un singolo ETF, tra cui:
    -   Rendimento totale e annualizzato (CAGR)
    -   Volatilità e massimo drawdown
    -   Sharpe, Sortino e Omega ratio
    -   Value at Risk (VaR)
-   **Confronto Multi-ETF**: Confronta le performance di più ETF o azioni su diversi orizzonti temporali, con grafici interattivi e normalizzati.
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
-   **Verifica Momentum**: Analizza il momentum di un elenco di ETF, correggendolo per il rischio e fornendo segnali operativi.

## Struttura del Progetto

etf_metrics_refactor/
├── app.py                   # Entrypoint per l'avvio dell'applicazione Streamlit
├── requirements.txt         # Dipendenze del progetto
├── README.md                # Questo file
└── etf_metrics/
├── init.py
├── benchmark.py         # Logica per l'inferenza del benchmark di un ETF
├── base_client.py       # Classe base per i client esterni con gestione errori e retry
├── config.py            # Costanti, configurazioni e portafogli modello
├── etf_info.py          # Recupero di informazioni estese su un ETF
├── etf_search_engine.py # Logica di ricerca e filtraggio degli ETF
├── justetf_client.py    # Client per il recupero dati da justETF
├── metrics.py           # Funzioni per il calcolo delle metriche di performance e rischio
├── momentum.py          # Logica per l'analisi del momentum
├── pac_screener.py      # Logica per lo screener tattico PAC
├── pipeline.py          # Orchestrazione dei calcoli per l'analisi di un singolo ETF
├── portfolio_backtester.py # Logica per il backtesting di portafogli
├── portfolio_tracker.py # Logica per il tracciamento di portafogli
├── trackingdiff_client.py # Client per il recupero della tracking difference
├── ui.py                # Gestione della navigazione principale e dello stato della sessione
├── ui_etf_comparison_app.py # UI per il confronto di ETF
├── ui_momentum_app.py   # UI per l'analisi del momentum
├── ui_pac_screener_app.py # UI per lo screener tattico PAC
├── ui_portfolio_backtester_app.py # UI per il backtesting di portafogli
├── ui_portfolio_tracker_app.py # UI per il tracciamento di portafogli
├── ui_single_etf.py     # UI per l'analisi di un singolo ETF
├── utils.py             # Funzioni di utilità generiche
└── yahoo_client.py      # Client per il recupero dati da Yahoo Finance


## Requisiti

Per l'elenco completo delle dipendenze, si veda il file `requirements.txt`. È consigliato l'uso di **Python ≥ 3.9**.

## Setup Rapido

1.  **Crea un ambiente virtuale:**

    ```bash
    python -m venv .venv
    ```

2.  **Attiva l'ambiente virtuale:**

    -   Su Windows:
        ```bash
        .venv\Scripts\activate
        ```
    -   Su macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Installa le dipendenze:**

    ```bash
    pip install -r requirements.txt
    ```

## Avvio

Per avviare l'applicazione, esegui il seguente comando dalla root del progetto:

```bash
streamlit run app.py