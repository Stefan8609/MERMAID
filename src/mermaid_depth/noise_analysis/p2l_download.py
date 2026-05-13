from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess


out_dir = Path("./WaveWatch_Data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 3

months = []

for year in [2018, 2019]:
    for month in range(1, 13):
        if year == 2018 and month < 6:
            continue

        months.append((year, month))


def download_month(year, month):
    ym = f"{year}{month:02d}"

    filename = f"LOPS_WW3-GLOB-30M_{ym}_p2l.nc"

    url = (
        "ftp://ftp.ifremer.fr/ifremer/dataref/ww3/"
        "GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/"
        f"{year}/FIELD_NC/{filename}"
    )

    outfile = out_dir / filename

    if outfile.exists():
        print(f"[{ym}] already exists, skipping")
        return

    print(f"[{ym}] downloading")

    subprocess.run(
        [
            "curl",
            "--fail",
            "-C",
            "-",
            "-o",
            str(outfile),
            url,
        ],
        check=True,
    )

    print(f"[{ym}] done")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []

    for year, month in months:
        futures.append(executor.submit(download_month, year, month))

    for future in futures:
        future.result()

print("\nDone.")