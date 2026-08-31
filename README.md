# 🏔️ Alps Explorer

An interactive map of ski resorts and investment villages across the French & Italian Alps.

Single self-contained `index.html` — no build step, no server. Open the file or serve the folder.

## Features

- **43 ski resorts** with piste length, altitude range, vertical drop and official links
- **33 investment villages** with altitude, buy rating and property listing links
- Search across names, regions and descriptions — accent-insensitive, so `meribel` finds Méribel
- Browsable sidebar list; click any row to fly to that place and open its details
- Toggle resorts / villages independently, and filter by France or Italy
- Street, Terrain and Satellite basemaps
- Nearby pins group into numbered clusters; click to expand
- Responsive — the list becomes a drawer on small screens

## Coverage

France: Tarentaise (Val d'Isère, Tignes, Les Arcs, La Plagne, Les 3 Vallées), Chamonix valley,
Portes du Soleil, Grand Massif, Aravis, Oisans, Serre-Chevalier, Vercors, Belledonne and the
southern Alps (Isola 2000, Auron, Valberg).

Italy: Aosta Valley, Piedmont / Via Lattea, Monterosa, Lombardy (Livigno, Bormio) and the
Dolomites (Cortina, Val Gardena, Val di Fassa, Alta Badia).

## How to deploy on GitHub Pages

1. Push `index.html` to the repository
1. Go to **Settings → Pages**
1. Set Source to **Deploy from branch**, pick the branch and `/ (root)`
1. Save — the site is served at `https://<user>.github.io/<repo>/`

## Data sources

- Ski resort data: official resort sites, SkiResort.info, Wikipedia
- Property links: Immobiliare.it, Idealista, SeLoger, LeBonCoin, Green Acres, Rightmove, Properstar
- Basemaps: Esri World Street Map, Esri World Topo Map, Esri World Imagery
- Map library: [Leaflet](https://leafletjs.com/) with [Leaflet.markercluster](https://github.com/Leaflet/Leaflet.markercluster)

Commune codes in French property links are INSEE codes, verified against `geo.api.gouv.fr`.
Several listing portals (SeLoger, LeBonCoin, Immobiliare, Idealista, Properstar) block
automated requests and will return 403 to a link checker while working normally in a browser.
