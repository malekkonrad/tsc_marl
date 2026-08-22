# deploy/hpc — ścieżka zapasowa (Cyfronet / Apptainer / SLURM)

**To NIE jest główna ścieżka uruchomieniowa.** Treningi liczone są na lokalnej maszynie
z GPU w venvie uv (patrz README w korzeniu repo). Ten katalog jest utrzymywany na
wypadek, gdyby zabrakło mocy — wtedy PLGrid/Cyfronet wraca do gry.

Uwaga o wersjach: obraz pinuje `torch 2.3.1 (cu121)`, a środowisko, w którym powstały
wyniki Fazy 0, to `torch 2.4.1`. Przed użyciem tej ścieżki do liczenia czegokolwiek
raportowalnego trzeba te piny zrównać z `pyproject.toml`, inaczej wyniki nie będą
porównywalne z resztą tabeli.

Build context to **korzeń repo**, nie ten katalog:

```bash
docker build -f deploy/hpc/Dockerfile -t tsc-marl:latest .
make hpc-build          # obraz -> tsc-marl.sif
./deploy/hpc/sync_to_cyfronet.sh push
```

Pułapki Cyfronetu (sufiks konta w `sbatch -A`, auto-czyszczenie `$SCRATCH`, Athena
tylko GPU, Helios na ARM) są opisane w historii tego pliku — patrz README z commita
`4ff2b0e`, sekcja 3.4.
