# 📒 Registro delle Strategie (Strategy Ledger)

> **File di cronaca di tutte le strategie del motore di backtest: attive, ritirate
> ed esperimenti.** Ogni volta che una strategia viene aggiunta, promossa o
> pensionata, questo registro va aggiornato con numeri e motivi.
>
> Ultimo aggiornamento: **2026-10-12**

---

## Come funziona il registro (dal punto di vista del codice)

| Dove | Cosa contiene |
|---|---|
| `etf_metrics/core/automated_backtest.py` → `STRATEGIES` / `STRATEGY_ORDER` | Le strategie **attive**: compaiono in Confronto Strategie, backtest singolo e Portafoglio Live |
| `etf_metrics/core/retired_strategies.py` → `RETIRED` | Le strategie **ritirate**, con codice integro, motivo e "lineage" (dove vive oggi l'idea) |
| `docs/STRATEGY_LEDGER.md` (questo file) | La **cronaca**: numeri, decisioni, criteri, lezioni |

**Riattivare una strategia ritirata** (per confronti sperimentali):

```python
from etf_metrics.core.retired_strategies import activate
activate("bear_market")      # da qui in poi ricompare in Confronto Strategie
```

---

## Metodologia di valutazione

- **Dati:** DB Xetra reale caricato da `market_data_parquet/` (nessuna chiamata a
  Yahoo), igienizzato da errori di livello prezzo (`_sanitize_ohlcv`: blocchi stub,
  tick corrotti, tripli/dimezzamenti in una seduta). **Il DB non viene mai modificato.**
- **Universo:** tutti i ticker con ≥ 200 righe (3.845 ticker, come il selettore
  "Tutto il DB" della UI).
- **Finestra intera:** 2021-01-04 → 2026-10-09, capitale 10.000 €, commissione 2 €,
  max 3 posizioni (parametri identici per tutte: confronto equo).
- **Split di robustezza (anti-overfitting):**
  - **Split A** = 2021-01 → 2023-12 (include il bear del 2022)
  - **Split B** = 2024-01 → 2026-10 (bull estremo)
  - Una strategia che rende solo in uno dei due split non è robusta.
- **Rendimenti per anno civile:** visibili nella UI (Confronto Strategie →
  "📅 Rendimenti per anno civile") e riportati qui sotto.

### ⚠️ Limiti dichiarati (da leggere sempre)

1. **Survivorship bias:** il DB contiene solo i ticker quotati *oggi*. Le strategie
   momentum che selezionano i vincitori passati trovano "gratis" i vincitori futuri
   perché i perdenti delistati mancano. I rendimenti assoluti sono **gonfiati**:
   usa il **confronto relativo** tra strategie e la **coerenza tra anni**, non i
   numeri in sé. (Esempio misurato: strategie +270-856% contro +61% dell'universo
   equal-weight sullo stesso periodo.)
2. **Significatività statistica:** sotto ~100 trade il risultato può essere fortuna.

### Il "bar di mantenimento" (criteri per restare attive)

| # | Criterio | Soglia |
|---|---|---|
| C1 | Robustezza per regime | Split A sopra l'Equal Weight B&H (+16.3%) e **max 2 anni civili negativi** su 6 |
| C2 | Rischio | MaxDD sull'intera finestra non peggiore di **-45%** |
| C3 | Significatività | **≥ 100 trade** sull'intera finestra |
| C4 | Non ridondanza | Se l'idea è già coperta da una strategia attiva con risultati migliori, si ritira |

**Esenzione documentata:** `golden_chandelier` è uno *strumento di misura* (replica
fedele della logica live di `trading.py`): serve a quantificare il rischio della
logica realmente in portafoglio, non a competere per il bar. Restare sotto -45% di
MaxDD è la sua diagnosi, non un motivo di ritiro.

> **Nota di onestà metodologica:** il bar è stato formalizzato il 2026-10-12 a valle
> della valutazione delle prime 7 strategie; i due esperimenti dello stesso giorno
> sono stati giudicati con questi criteri. Dalle prossime valutazioni il bar va
> dichiarato **prima** di eseguire il backtest (pre-registrazione), per evitare di
> razionalizzare a posteriori.

### Come rieseguire i numeri (offline, solo DB locale)

```bash
cd /home/coder/portfolio_tracker
PYTHONPATH=. python3 scripts/run_xetra_comparison.py                    # finestra intera
PYTHONPATH=. python3 scripts/run_xetra_comparison.py <keys> 2021-01-04 2023-12-29  # split A
PYTHONPATH=. python3 scripts/run_xetra_comparison.py <keys> 2024-01-02              # split B
```

---

## ✅ Strategie ATTIVE (6)

### Numeri — finestra intera 2021-01-04 → 2026-10-09

| Strategia | Rend.% | CAGR% | MaxDD% | Sharpe | Trades | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *S&P500 B&H* | *111.1* | — | — | — | 0 | *28.8* | *-19.5* | *24.3* | *23.3* | *16.4* | *14.2* |
| *Equal weight B&H* | *61.3* | — | — | — | 0 | *21.4* | *-13.3* | *10.4* | *11.0* | *13.6* | *10.0* |
| 🧠 AI Momentum | 855.9 | 48.0 | -32.3 | 1.1 | 480 | 28.0 | 22.8 | 18.6 | 87.6 | 88.4 | 45.0 |
| 🌊 Trend Fusion | 405.8 | 32.5 | -30.9 | 0.9 | 396 | 38.5 | -4.2 | 41.9 | 71.7 | 13.2 | 38.5 |
| 🐢 Turtle Breakout | 270.0 | 25.5 | -29.6 | 0.9 | 274 | 17.0 | 5.2 | 9.6 | 73.7 | 9.3 | 44.7 |
| 🛡️ Golden Chandelier *(strumento di misura)* | 496.6 | 36.4 | -47.9 | 0.9 | 362 | 40.1 | -7.3 | 27.9 | 116.4 | -27.7 | 129.9 |
| 🚦 Golden Chandelier Difensivo | 441.7 | 34.1 | -34.7 | 1.0 | 358 | 57.0 | 6.2 | 51.0 | 55.6 | -17.0 | 66.8 |
| 🧘 Steady Winners (low-vol) | 149.7 | 17.2 | -12.7 | 0.7 | 176 | 49.9 | 5.8 | 13.4 | 2.8 | 22.6 | 10.3 |

### Numeri — split di robustezza

| Strategia | Split A 21-23 (bear incluso) | MaxDD A | Sharpe A | Split B 24-26 (bull estremo) | MaxDD B | Sharpe B |
|---|---:|---:|---:|---:|---:|---:|
| *S&P500 B&H* | *+28.9%* | — | — | *+64.7%* | — | — |
| *Equal weight B&H* | *+16.3%* | — | — | *+38.9%* | — | — |
| 🧠 AI Momentum | +91.3% | -25.6 | 0.7 | +280.3% | -38.3 | 1.2 |
| 🌊 Trend Fusion | +88.8% | -23.8 | 0.8 | +1478.9% | -29.2 | 2.0 |
| 🐢 Turtle Breakout | +35.4% | -19.4 | 0.4 | +880.6% | -20.2 | 1.8 |
| 🛡️ Golden Chandelier | +68.8% | -21.5 | 0.6 | +269.6% | -42.4 | 1.2 |
| 🚦 Golden Chandelier Difensivo | **+181.6%** | **-11.2** | **1.6** | +44.7% | -27.8 | 0.4 |
| 🧘 Steady Winners (low-vol) | +80.6% | -12.7 | 0.8 | +59.5% | **-9.2** | 1.0 |

### Dossier

**🧠 AI Enhanced Momentum** (`ai_momentum`) — *flagship, strategia originale.*
RS 63gg + ADX>25 + prossimità massimi 52s, ingressi solo in BULL, stop ATR 2.5x,
TP1 +12% con lock a breakeven, time stop 21gg.
**Perché attiva:** unica con **tutti e 6 gli anni civili positivi** (anche il 2022:
+22.8% contro S&P500 -19.5%); miglior Sharpe della suite (1.1); 480 trade =
significatività piena. C1 ✓ C2 ✓ C3 ✓ C4 ✓.

**🌊 Trend Fusion** (`trend_fusion`) — fusione dei concetti migliori: breakout
Donchian 55 **in congiunzione** con Value Area High + volume ≥1.5x + ADX/RS,
TP parziale con lock, uscita LL20 che lascia correre il trend.
**Perché attiva:** unico anno negativo il 2022 (-4.2%), MaxDD contenuto (-30.9%),
rendimento pieno in entrambi gli split (+88.8% / +1478.9%). C1 ✓ C2 ✓ C3 ✓ C4 ✓.
*(Assorbe le idee di bear_market e volume_profile, congiunte in modo più prudente.)*

**🐢 Turtle Breakout** (`turtle_breakout`) — Donchian 55/20 stile Turtle, classico
senza fronzoli.
**Perché attiva:** zero anni negativi, MaxDD -29.6% (migliore tra le strategie
aggressive), semplice e robusta per costruzione (parametri storici, non tarati).
C1 ✓ C2 ✓ C3 ✓ C4 ✓.

**🛡️ Golden Chandelier** (`golden_chandelier`) — *strumento di misura, esente dal bar.*
Replica 1:1 della logica live delle posizioni satellite (`trading.py`: SMA130 +
momentum 6M, Chandelier Exit con scala `atr_multiplier_from_vol`).
**Perché attiva:** serve a **misurare** la logica realmente in portafoglio. Diagnosi
al 2026-10-12: rende (+496.6%) ma con MaxDD **-47.9%**, anni negativi 2022 (-7.3%)
e 2025 (-27.7%), perché **non ha filtro macro** e compra anche in bear.

**🚦 Golden Chandelier Difensivo** (`golden_chandelier_def`) — *esperimento
promosso il 2026-10-12.*
Identica alla logica live ma con il pacchetto macro di AI Momentum (acquisti solo
in BULL + uscita d'emergenza in crisi). **Esperimento a variabile singola** per
isolare l'effetto del filtro.
**Esito:** ipotesi confermata — il filtro macro taglia il MaxDD da -47.9% a
**-34.7%**, ribalta il 2022 (-7.3% → **+6.2%**) e nello split A è la migliore in
assoluto (+181.6%, Sharpe 1.6, MaxDD -11.2%). Costo: -55pp di rendimento sulla
finestra intera, per lo più nel bull estremo 2024-26 (55.6% vs 116.4% nel 2024).
**Raccomandazione pratica:** aggiungere il filtro macro alla logica live di
`trading.py` — oppure gestire il live con questa strategia nel Portafoglio Live.
C1 ✓ C2 ✓ C3 ✓ (358) C4 ✓.

**🧘 Steady Winners** (`low_vol_trend`) — *esperimento promosso il 2026-10-12.*
Test dell'**anomalia low-volatility**: stessi ingressi della logica live ma solo su
titoli con volatilità annualizzata ≤ 1.25× il benchmark (i vincitori *calmi*, non i
pump). Uscite identiche (chandelier + SMA130).
**Esito:** l'anomalia regge su Xetra — **tutti e 6 gli anni positivi**, MaxDD
**-12.7%** (il migliore dell'intera suite, benchmark incluso), +149.7% totale
contro +61.3% dell'universo equal-weight con un terzo del rischio. Split B
da manuale: +59.5% con MaxDD -9.2% e Sharpe 1.0.
**Ruolo:** estremo difensivo del fronte rischio/rendimento della suite.
C1 ✓ C2 ✓✓ C3 ✓ (176) C4 ✓ (profilo unico, non coperto da nessun'altra).

---

## 🗑️ Strategie RITIRATE (5)

### Ritirate il 2026-10-12 (da questa sessione di valutazione)

**🐻 Bear Market Regime Switcher** (`bear_market`) — *codice in
`retired_strategies.py`, riattivabile con `activate("bear_market")`.*
Comprava i forti in BULL e cercava ETF inversi/oro in BEAR, cash nei regimi di crisi.
Numeri al congedo: +282.9% totale, **MaxDD -69.2%** (il peggiore), Sharpe 0.3,
296 trade; split A +123.3% (era la migliore lì) ma split B +120.9% con MaxDD -56.2%
e 2026 -20.5%.
**Motivo (C2 violato):** il peggior profilo di rischio della suite. L'idea di
comprare chi sale mentre il mercato scende funziona nel bear pulito del 2022
(+11.2%) ma nel regime whipsaw 2025-26 fa sanguinare inseguendo prodotti inversi
che decadono.
**Eredità:** lo switch difensivo vive in `trend_fusion` (congiunzioni prudenti) e
nel filtro macro di `golden_chandelier_def`.

**📊 Volume Profile / Value Area Breakout** (`volume_profile`) — *codice in
`retired_strategies.py`, riattivabile con `activate("volume_profile")`.*
Breakout sopra il Value Area High (profilo del volume su 60gg) con conferma di
volume ≥1.5x e prezzo sopra SMA50; uscita al rientro sotto il POC.
Numeri al congedo: +312.1% totale, MaxDD -37.7%, Sharpe 0.7, 408 trade; **split A
-2.8%** (unica sotto pari), anni negativi nel 2022, 2023 e 2025.
**Motivo (C1 e C4 violati):** la meno coerente — tre anni negativi in tre regimi
diversi. Il segnale VAH+volume da solo compra falsi breakout: senza conferma di
trend (ADX/RS) non basta.
**Eredità:** il breakout dal Value Area con volume è uno dei tre ingredienti di
congiunzione di `trend_fusion`, dove rende perché *richiesto insieme* agli altri.

**🔄 Dual Momentum 12-1** (`dual_momentum`, versione 2) — *codice in
`retired_strategies.py`, riattivabile con `activate("dual_momentum")`.*
Rotazione mensile in stile Antonacci: momentum 12 mesi saltando l'ultimo (12-1),
filtri assoluti SMA200 + momentum positivo, uscite di segnale, nessun trailing.
Parametri standard di letteratura, nessun tuning.
Numeri al congedo: +392.0% totale, Sharpe 0.3, **78 trade**; split A +4.5% (sotto
l'equal-weight +16.3%), 2022 -16.3%, e **+267.5% nel solo 2026** (due terzi del
rendimento totale in un anno).
**Motivo (C1 e C3 violati):** l'implementazione è corretta ma su questo universo
l'idea non paga: nel periodo difficile sottoperforma l'universo medio e il risultato
complessivo poggia sulla concentrazione su pochi sopravvissuti del bias, non su
robustezza.
**Eredità / futuro:** da rivalutare su finestre più lunghe, asset class globali o
universi meno distorti dal survivorship bias. Storia curiosa: la **v1** (momentum
252gg, 2 posizioni, uscita su flip) era stata ritirata il 2026-10-05 nel commit
`191baea`; la v2 è nata e ritirata il 2026-10-12 con motivazione quantitativa.

### Rimosse il 2026-10-05 (commit `191baea` "feat: new algorithm feature")

> Cronaca ricostruita dalla storia git: in quel commit la suite fu ripulita e nacque
> `trend_fusion`, che assorbì i concetti migliori. Numeri d'epoca non disponibili
> (il confronto attuale su dati igienizzati non esisteva ancora).

**🌊 Order Flow — CVD & Accumulo** (`orderflow`). Stimava la pressione compratrice
dalle candele daily (Volume × posizione del close nel range H-L), accumulava il CVD
e comprava dopo ≥3 giorni di accumulo su 10, sopra SMA50 con RS positivo. Uscita in
distribuzione (CVD sotto la media per 3 giorni). *Rimossa nella pulizia del
2026-10-05.*

**🎯 Mean Reversion — RSI-2 Pullback** (`mean_reversion`). Comprava i panic-dip
(RSI a 2 giorni < 10) solo in trend lungo (Close > SMA200), uscita al ripristino
(RSI-2 > 80 o close sopra SMA20), stop ATR 2.5x e time stop 10 giorni. *Rimossa
nella pulizia del 2026-10-05. Nota: idea da rivalutare con dati migliori — è lo
stile complementare al trend following e al momento la suite non ne ha.*

**🔄 Dual Momentum v1** (`dual_momentum`, prima versione). Momentum a 252 giorni
senza salto dell'ultimo mese, max 2 posizioni, uscita al flip del momentum, stop
di sicurezza 5x ATR. *Rimossa il 2026-10-05; reincarnata come v2 (12-1 Antonacci)
il 2026-10-12 e di nuovo ritirata lo stesso giorno: vedi sopra.*

---

## 📅 Cronologia

| Data | Evento | Dettaglio |
|---|---|---|
| ≤ 2026-10-02 (`bc9e495`) | Prima istantanea nel repo | 7 strategie: `ai_momentum`, `bear_market`, `volume_profile`, `turtle_breakout`, `dual_momentum` v1, `orderflow`, `mean_reversion` |
| 2026-10-05 (`191baea`) | Pulizia della suite | − `dual_momentum` v1, − `orderflow`, − `mean_reversion`; + `trend_fusion` (fusione dei concetti migliori superstiti) |
| 2026-10-12 | Coerenza live/backtest | + `golden_chandelier` (replica backtest della logica live di `trading.py`) |
| 2026-10-12 | Esperimento letteratura | + `dual_momentum` v2 (12-1 Antonacci, parametri di letteratura) |
| 2026-10-12 | Esperimenti a variabile singola | + `golden_chandelier_def` (filtro macro sulla logica live) e + `low_vol_trend` (anomalia low-vol): **entrambi promossi** |
| 2026-10-12 | PENSIONAMENTI | − `bear_market` (MaxDD -69.2%), − `volume_profile` (split A sotto pari), − `dual_momentum` v2 (fragile, 78 trade); nasce questo registro e `retired_strategies.py` |

---

## 📋 Procedura per il futuro

### Aggiungere una strategia

1. **Ipotizza prima:** quale effetto di mercato sfrutta? Perché *dovrebbe*
   funzionare? (letteratura, meccanica, non curve-fitting)
2. Implementala in `automated_backtest.py` (sottoclasse di `BaseStrategy`; se è una
   variante di una esistente, eredita e cambia UNA cosa sola).
3. Aggiungi chiave a `STRATEGIES` / `STRATEGY_ORDER` e una voce in questo registro.
4. Valuta con lo script di confronto: finestra intera + split A/B + anni civili.
5. Giudica col bar (C1-C4), dichiara l'esito qui (promossa o ritirata).

### Ritirare una strategia

1. Sposta la classe in `etf_metrics/core/retired_strategies.py` con intestazione
   "RITIRATA <data>" e aggiungila al dict `RETIRED` (motivo + lineage).
2. Rimuovila da `STRATEGIES` / `STRATEGY_ORDER`.
3. Aggiorna questo registro: numeri al congedo, motivo, eredità, riga in cronologia.

### Promuovere una ritirata

`activate(key)` + voce nel registro che documenti *perché* ora funziona
(dat migliori? regime cambiato? universo ricostruito con i delistati?).

---

## 🧠 Lezioni (post-mortem accumulate)

1. **Il filtro macro è il miglioramento singolo più grande** mai misurato in questa
   suite: sulla stessa logica live, MaxDD da -47.9% a -34.7%, 2022 da -7.3% a +6.2%,
   split A da +68.8% a +181.6% con Sharpe 1.6 (`golden_chandelier` vs
   `golden_chandelier_def`).
2. **L'anomalia low-volatility è reale anche qui:** -12.7% di MaxDD, sei anni
   positivi su sei, doppio dell'universo equal-weight con un terzo del rischio.
3. **Inseguire i vincitori caldi paga solo nel bias:** i numeri da +880% a +1479%
   dello split 2024-26 sono gonfiati dal survivorship bias (sopravvissuti estremi
   come i pump small-cap del 2025-26). Non sono ripetibili in un universo onesto.
4. **I prodotti inversi in regime whipsaw dissanguano** (`bear_market`): ok nel
   bear pulito 2022, disastrosi nei falsi bear 2025-26.
5. **Un segnale di prezzo/volume senza conferma di trend compra falsi breakout**
   (`volume_profile`): la stessa condizione, in *congiunzione* con ADX/RS, funziona
   (dentro `trend_fusion`).
6. **Un anno che porta tutto il rendimento è fragilità, non bontà**
   (`dual_momentum`: +267.5% nel 2026 = due terzi del totale). Guarda sempre la
   tabella per anno civile prima di innamorarti di un numero totale.
7. **Confronta sempre con l'equal-weight dell'universo**, non solo col S&P500: è
   il riferimento onesto per capire quanto del rendimento è *selezione* e quanto è
   *bias dell'universo*.
