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
-   **Trading & Backtest**: Scanner di segnali operativi e backtest di una strategia algoritmica trend-following (filtro SMA 130 + momentum a 6 mesi), con report di rischio (Sharpe, Sortino, Max Drawdown, Profit Factor).
-   **Portafoglio Live**: Gestione operativa di un portafoglio algoritmico con stop loss / take profit trailing, sizing dinamico e regime di mercato.

## Struttura del Progetto

```
portfolio_tracker/
├── app.py                          # Entrypoint per l'avvio dell'applicazione Streamlit
├── pyproject.toml                  # Dipendenze e configurazione del progetto (Poetry)
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
    │   ├── data_manager.py         # Cache locale (SQLite) delle serie storiche
    │   └── metrics.py              # Funzioni per le metriche di performance e rischio
    ├── clients/
    │   ├── base_client.py          # Classe base per i client esterni (retry, errori)
    │   └── yahoo_client.py         # Client per il recupero dati da Yahoo Finance
    └── shared/
        ├── config.py               # Costanti, configurazioni e portafogli modello
        └── utils.py                # Funzioni di utilità generiche
```

## Requisiti

Per l'elenco completo delle dipendenze, si veda il file `pyproject.toml`. È consigliato l'uso di **Python ≥ 3.9**.

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
    pip install -r <(poetry export -f requirements.txt)
    ```

    oppure, con Poetry:

    ```bash
    poetry install
    ```

## Avvio

Per avviare l'applicazione, esegui il seguente comando dalla root del progetto:

```bash
streamlit run app.py
```
