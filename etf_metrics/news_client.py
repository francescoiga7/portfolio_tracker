# -*- coding: utf-8 -*-
from typing import List, Dict, Any

# Importiamo direttamente lo strumento di ricerca disponibile
# Nota: questo import è simbolico per la logica del modello.
# L'effettiva chiamata avverrà tramite tool_code.
import google_search


def search_google_news(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Esegue una ricerca di notizie tramite lo strumento google_search e
    restituisce i risultati in un formato strutturato.

    Args:
        query (str): La stringa di ricerca per le notizie.
        num_results (int): Il numero massimo di risultati da restituire.

    Returns:
        List[Dict[str, Any]]: Una lista di dizionari, ogni dizionario rappresenta una notizia.
    """
    try:
        # Questa è la chiamata effettiva che il modello eseguirà come tool_code
        search_results = google_search.search(queries=[query])

        if not search_results or not search_results[0].results:
            return []

        formatted_results = []
        for res in search_results[0].results[:num_results]:
            source_domain = "N/D"
            if res.url:
                try:
                    # Estrae il dominio dall'URL per avere una fonte pulita
                    source_domain = res.url.split('//')[1].split('/')[0].replace('www.', '')
                except IndexError:
                    pass

            formatted_results.append({
                "title": res.source_title,
                "snippet": res.snippet,
                "link": res.url,
                "source": source_domain,
                "publication_time": res.publication_time
            })

        return formatted_results

    except Exception as e:
        print(f"Errore durante l'esecuzione di google_search: {e}")
        return []