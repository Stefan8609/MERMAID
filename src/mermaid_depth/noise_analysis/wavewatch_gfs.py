from herbie import Herbie
import pandas as pd
import matplotlib.pyplot as plt


def to_utc_naive(time):
    """
    Convert input time to UTC, then remove timezone info.
    Herbie expects timezone-naive UTC timestamps.
    """
    ts = pd.to_datetime(time, utc=True)
    return ts.tz_convert(None)


def gfswave_field(
    valid_time,
    product="epacif.0p16",
    variable_regex=":HTSGW:",
):
    valid_time = to_utc_naive(valid_time)

    cycle_hour = (valid_time.hour // 6) * 6
    cycle_time = valid_time.normalize() + pd.Timedelta(hours=cycle_hour)
    fxx = int((valid_time - cycle_time) / pd.Timedelta(hours=1))

    H = Herbie(
        cycle_time,
        model="gfs_wave",
        product=product,
        fxx=fxx,
    )

    ds = H.xarray(variable_regex, remove_grib=False)

    ds = ds.assign_attrs(
        requested_valid_time=str(valid_time),
        cycle_time=str(cycle_time),
        fxx=fxx,
        product=product,
        variable_regex=variable_regex,
    )

    return ds


if __name__ == "__main__":
    point = gfswave_field(
        valid_time="2022-06-15 14:00",
        product="epacif.0p16",
        variable_regex=":HTSGW:",
    )

    print(point)