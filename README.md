
# ETF Metrics & Tracking Difference — Refactor (solo Streamlit)

Refactor modulare dell'app **ETF Metrics** con rimozione della CLI e avvio esclusivo tramite **Streamlit**. Include:

- Calcolo metriche (Totale, CAGR, Vol ann, Max DD)
- TER da justETF (fallback Yahoo `info`)
- Benchmark inferito/override e **Tracking Difference** (TD) su più periodi
- **TD rolling 12 mesi** (~252 sedute)
- KPI extra: **Sharpe ratio** (con RF), **Drawdown corrente**
- Confronto multi‑ETF (grafici sempre in %)

## Struttura del progetto
```
etf_metrics_refactor/
├── app.py                   # entrypoint Streamlit
├── requirements.txt
├── README.md
└── etf_metrics/
    ├── __init__.py
    ├── config.py            # costanti e configurazione
    ├── utils.py             # utility generiche (formattazione, % index, ticker picking)
    ├── yahoo_client.py      # ricerca Yahoo e download serie storiche
    ├── metrics.py           # metriche core (CAGR, vol, MDD, Sharpe, DD corr.)
    ├── justetf_client.py    # fetch & parsing pagine justETF (TER, benchmark name)
    ├── benchmark.py         # inferenza proxy benchmark
    ├── ter.py               # logica TER (justETF + fallback)
    ├── tracking.py          # TD rolling 12m + fetch da trackingdifferences.com
    ├── io_csv.py            # salvataggio/upsert CSV risultati
    └── pipeline.py          # orchestrazione calcoli per più periodi
```

## Requisiti
Vedi `requirements.txt`. Suggerita **Python ≥ 3.9**.

## Setup rapido
```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Avvio
```bash
streamlit run app.py
```

## Note
- Per un FTSE All‑World puoi usare l'**ISIN** `IE00BK5BQT80`.
- Il benchmark viene inferito automaticamente (con possibilità di **override** nella sidebar).
- Il file di output CSV viene aggiornato in modalità *upsert* per chiave `(isin, period)`.

## Sicurezza e limiti
- Dati di mercato via Yahoo Finance (yfinance); possibili limitazioni/ritardi.
- Il parsing HTML di justETF e trackingdifferences può rompersi se cambiano i layout.

## Licenza
MIT (se non diversamente specificato).
