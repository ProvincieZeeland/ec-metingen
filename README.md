# Download KNMI Neerslag Rasters

Dit script downloadt neerslagdata van het KNMI, verwerkt de bestanden, zet ze om naar GeoTIFF en slaat ze op in een database.

## Installatie

1. Zorg ervoor dat je Python (3.13 of hoger) geïnstalleerd hebt.
2. Installeer de benodigde afhankelijkheden:
   ```
   pip install -r requirements.txt
   ```
3. Maak een configuratiebestand `config.ini` aan met de volgende structuur:
   ```
   [api]
   key=JOUW_API_KEY

   [paths]
   download_directory=downloads
   output_directory=output

   [dates]
   start_date=2023-01-01

   [database]
   host=localhost
   port=5432
   user=gebruiker
   password=wachtwoord
   dbname=database
   schema=public
   source_layer=bronlaag
   new_layer=nieuwe_laag
   geometry_column=geom
   ```

## Gebruik

1. Start het script met:
   ```
   python Download_KNMI_Neerslag_rasters.py
   ```
2. Het script zal:
   - De meest recente neerslagdata downloaden uit de KNMI API.
   - De bestanden filteren op basis van een datumbereik.
   - NetCDF-bestanden omzetten naar GeoTIFF.
   - De gegevens analyseren en verwerken.
   - Resultaten opslaan in een PostgreSQL/PostGIS database.

## Functionaliteiten

- **Dataset Download**: Ophalen van neerslagdata via de KNMI API.
- **Bestandsbeheer**: Controle of bestanden al gedownload zijn.
- **Conversie**: NetCDF-bestanden worden geconverteerd naar GeoTIFF.
- **Data-analyse**: Berekenen van neerslagwaarden per locatie.
- **Database opslag**: Invoeren van gegevens in PostGIS-tabellen.

## Vereisten

- Python 3.13+
- PostgreSQL met PostGIS-extensie
- KNMI API-sleutel

## Problemen oplossen

- **Geen verbinding met database?** Controleer de databaseconfiguratie in `config.ini`.
- **Geen bestanden gedownload?** Controleer de API-sleutel en startdatum.
- **GeoTIFF niet aangemaakt?** Zorg ervoor dat `xarray` en `rioxarray` correct zijn geïnstalleerd.

