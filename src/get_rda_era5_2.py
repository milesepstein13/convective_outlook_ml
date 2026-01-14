import dataclasses
from typing import List, Dict, Tuple, Union, Optional
import datetime as dt
from dateutil.relativedelta import relativedelta
import xarray as xr
import warnings
import concurrent.futures

LEVEL_DATA_PATH = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
SFC_DATA_PATH   = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"

CHANNEL_TO_CODE = {
    "10u": 165, "10v": 166, "2t": 167, "2d": 168, "z": 129, "tisr": 212,
    "pv": 60, "q": 133, "t": 130, "u": 131, "v": 132, "w": 135,
}
SURFACE_CHANNELS = ["10u", "10v", "2t", "2d", "z", "tisr"]

@dataclasses.dataclass(frozen=True)
class PressureVarRequest:
    short: str          # e.g. "t"
    code_id: int        # e.g. 130
    levels: Tuple[int]  # e.g. (925,850,700,500,300)

@dataclasses.dataclass(frozen=True)
class SurfaceVarRequest:
    short: str          # e.g. "2t"
    code_id: int        # e.g. 167
    code0: int = 128

def _termll025(var_short: str) -> str:
    return "ll025uv" if var_short in ("u", "v") else "ll025sc"

def _month_end_day(time: dt.datetime) -> int:
    return (time + relativedelta(day=31)).day

def _infer_var_name(ds: xr.Dataset) -> str:
    keys = list(ds.data_vars.keys())
    if not keys:
        raise ValueError("No data variables found in dataset.")
    if keys[0] != "utc_date":
        return keys[0]
    if len(keys) < 2:
        raise ValueError("Only utc_date present; cannot infer data variable.")
    warnings.warn(ResourceWarning(f"Please check var name {keys[1]}!"))
    return keys[1]

def _surface_path(req: SurfaceVarRequest, time: dt.datetime) -> str:
    year = f"{time.year:04d}"
    month = f"{time.month:02d}"
    end_day = _month_end_day(time)
    term = _termll025(req.short)
    return (
        f"{SFC_DATA_PATH}{year}{month}/"
        f"e5.oper.an.sfc.{req.code0}_{req.code_id}_{req.short}.{term}."
        f"{year}{month}0100_{year}{month}{end_day:02d}23.nc"
    )

def _pressure_path(req: PressureVarRequest, date: dt.date) -> str:
    year = f"{date.year:04d}"
    month = f"{date.month:02d}"
    day = f"{date.day:02d}"
    term = _termll025(req.short)
    return (
        f"{LEVEL_DATA_PATH}{year}{month}/"
        f"e5.oper.an.pl.128_{req.code_id}_{req.short}.{term}."
        f"{year}{month}{day}00_{year}{month}{day}23.nc"
    )

def _subset_space(da: xr.DataArray, lat_slice, lon_slice, space_thin: int) -> xr.DataArray:
    # native ERA5 names in these files: latitude/longitude
    da = da.sel(latitude=lat_slice, longitude=lon_slice)
    if space_thin and space_thin > 1:
        da = da.isel(latitude=slice(None, None, space_thin),
                     longitude=slice(None, None, space_thin))
    return da.rename({"latitude": "lat", "longitude": "lon"})

def _times_for_12z_day(day: dt.date, tods: List[int]) -> List[dt.datetime]:
    base = dt.datetime(day.year, day.month, day.day, 0)
    return [base + dt.timedelta(hours=int(t) + 12) for t in tods]

def _split_times_by_date(times: List[dt.datetime]) -> Dict[dt.date, List[dt.datetime]]:
    out: Dict[dt.date, List[dt.datetime]] = {}
    for t in times:
        out.setdefault(t.date(), []).append(t)
    return out

def _worker_read_surface(
    req: SurfaceVarRequest,
    times: List[dt.datetime],
    lat_slice, lon_slice, space_thin: int
) -> xr.DataArray:
    # NOTE: Surface files are monthly; to keep things simple and per your constraint,
    # we allow multiple opens if times span month boundaries.
    # Group by (year,month) so each monthly file opened once per request.
    groups: Dict[Tuple[int,int], List[dt.datetime]] = {}
    for t in times:
        groups.setdefault((t.year, t.month), []).append(t)

    parts = []
    for (yy, mm), tlist in sorted(groups.items()):
        path = _surface_path(req, dt.datetime(yy, mm, 1))
        ds = xr.open_dataset(path)
        var = _infer_var_name(ds)
        sub = ds[var].sel(time=tlist)
        sub = _subset_space(sub, lat_slice, lon_slice, space_thin)
        parts.append(sub)

    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    # dims: time, lat, lon
    return da

def _worker_read_pressure_dayfile(
    req: PressureVarRequest,
    date: dt.date,
    times_for_that_date: List[dt.datetime],
    lat_slice, lon_slice, space_thin: int
) -> xr.DataArray:
    # This is the key: open this PL file ONCE and pull all times and all levels.
    path = _pressure_path(req, date)
    ds = xr.open_dataset(path)
    var = _infer_var_name(ds)

    da = ds[var].sel(time=times_for_that_date, level=list(req.levels))
    # dims: time, level, latitude, longitude
    da = _subset_space(da, lat_slice, lon_slice, space_thin)
    # dims: time, level, lat, lon
    return da

def _parse_channels_to_requests(channels: List[str]) -> Tuple[List[SurfaceVarRequest], List[PressureVarRequest]]:
    surface: List[SurfaceVarRequest] = []
    pl_levels_by_var: Dict[str, List[int]] = {}

    for ch in channels:
        if ch in SURFACE_CHANNELS:
            surface.append(SurfaceVarRequest(short=ch, code_id=CHANNEL_TO_CODE[ch]))
        else:
            short = ch[:-3]         # e.g. "t"
            lvl = int(ch[-3:])      # e.g. 850
            pl_levels_by_var.setdefault(short, []).append(lvl)

    pressure: List[PressureVarRequest] = []
    for short, lvls in pl_levels_by_var.items():
        pressure.append(PressureVarRequest(short=short, code_id=CHANNEL_TO_CODE[short], levels=tuple(sorted(set(lvls)))))

    return surface, pressure

def _time_to_tod_map(day0: dt.date, tods: List[int]) -> Dict[dt.datetime, int]:
    times = _times_for_12z_day(day0, tods)
    return {t: int(tod) for t, tod in zip(times, tods)}

def get_12z_day_block(
    channels: List[str],
    day0: dt.date,
    tods: List[int],
    lat_slice,
    lon_slice,
    space_thin: int = 1,
    max_workers: int = 8,
) -> xr.DataArray:
    """
    Returns DataArray dims: (day=1, tod, channel, lat, lon)

    - PL files: opened once per (var, date) within this call.
    - Surface files: may open monthly files per surface var (allowed).
    """
    surface_reqs, pressure_reqs = _parse_channels_to_requests(channels)

    # absolute requested times for this 12Z–12Z day
    times_all = _times_for_12z_day(day0, tods)
    times_by_date = _split_times_by_date(times_all)
    t2tod = _time_to_tod_map(day0, tods)

    futures = []
    results_surface: Dict[str, xr.DataArray] = {}
    results_pressure: Dict[str, List[xr.DataArray]] = {req.short: [] for req in pressure_reqs}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        # surface: per req (may open 1-2 month files internally)
        for req in surface_reqs:
            fut = ex.submit(_worker_read_surface, req, times_all, lat_slice, lon_slice, space_thin)
            futures.append(("sfc", req.short, None, fut))

        # pressure: per (req, date) -> ensures each PL file opened once
        for req in pressure_reqs:
            for date, tlist in times_by_date.items():
                fut = ex.submit(_worker_read_pressure_dayfile, req, date, tlist, lat_slice, lon_slice, space_thin)
                futures.append(("pl", req.short, date, fut))


        for kind, short, date, fut in futures:
            da = fut.result()
            if kind == "sfc":
                results_surface[short] = da
            else:
                results_pressure[short].append(da)

    # ---- assemble into channel-stacked array (tod, channel, lat, lon)

    channel_blocks: List[xr.DataArray] = []

    # surface: da dims (time, lat, lon) -> remap time->tod, then expand channel
    for req in surface_reqs:
        da = results_surface[req.short]

        tod_coord = [t2tod[t] for t in list(da["time"].values.astype("datetime64[ns]").astype("datetime64[ms]").astype(object))]  

    # The above conversion is fragile across numpy/pandas; do it robustly:
    def _np64_to_py(dt64):
        # numpy datetime64 -> python datetime
        ts = xr.DataArray([dt64]).to_index()[0]
        return ts.to_pydatetime()

    # surface
    for req in surface_reqs:
        da = results_surface[req.short]
        tod_coord = [t2tod[_np64_to_py(t)] for t in da["time"].values]
        da = da.assign_coords(tod=("time", tod_coord)).swap_dims({"time": "tod"}).drop_vars("time")
        da = da.expand_dims(channel=[req.short])  # (channel, tod, lat, lon) after transpose
        da = da.transpose("tod", "channel", "lat", "lon")
        channel_blocks.append(da)

    # pressure: each short has 1-2 pieces (day0 and possibly day1) dims (time, level, lat, lon)
    for req in pressure_reqs:
        print(req)
        pieces = results_pressure[req.short]
        print(1)
        da = xr.concat(
            pieces,
            dim="time",
            coords="minimal",
            compat="override",
            join="override",
        )
        print(2)
        # remap time->tod
        tod_coord = [t2tod[_np64_to_py(t)] for t in da["time"].values]
        print(3)
        da = da.assign_coords(tod=("time", tod_coord)).swap_dims({"time": "tod"}).drop_vars("time")
        # da dims: tod, level, lat, lon
        print(4)
        # stack level into channel strings once, without reopening files
        # channel names match your existing scheme, e.g. t850, z500, u300
        ch_names = [f"{req.short}{int(lv):03d}" for lv in da["level"].values]
        print(5)
        da = da.assign_coords(channel=("level", ch_names)).swap_dims({"level": "channel"}).drop_vars("level")
        print(6)
        # da dims: tod, channel, lat, lon
        channel_blocks.append(da)

    out = xr.concat(channel_blocks, dim="channel")  # (tod, channel, lat, lon)

    # add day dim (length 1)
    out = out.expand_dims(day=[dt.datetime(day0.year, day0.month, day0.day)])
    out = out.transpose("day", "tod", "channel", "lat", "lon")
    return out


@dataclasses.dataclass
class ERA5DataSource:
    channel_names: List[str]

    def get_12z_day_block(
        self,
        day: dt.date,
        tods: List[int],
        lat_slice,
        lon_slice,
        space_thin: int = 1,
        max_workers: int = 8,
    ) -> xr.DataArray:
        return get_12z_day_block(
            channels=self.channel_names,
            day0=day,
            tods=tods,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
            space_thin=space_thin,
            max_workers=max_workers,
        )