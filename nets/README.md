# nets/

Put your simulation scenarios here.

- **SUMO**: `*.net.xml` + `*.rou.xml` (+ optional `*.sumocfg`, `*.poly.xml`).
  Generate a real city with `python $SUMO_HOME/tools/osmWebWizard.py`, or convert
  an OSM extract with `netconvert --osm-files city.osm.xml -o city.net.xml`.
- **CityFlow**: `roadnet.json` + `flow.json` + a top-level config JSON.
  Convert a SUMO net with `python /opt/CityFlow/tools/Converter/converter.py`.

Standardized benchmark scenarios (Cologne, Ingolstadt, grid4x4, arterial4x4)
already ship inside the image:
- sumo-rl nets: check the installed `sumo_rl/nets/RESCO` directory.
- RESCO harness + scenarios: `/opt/RESCO` inside the container.

Large generated networks are gitignored by pattern where noted; commit only the
scenario definitions you actually use so runs stay reproducible.
