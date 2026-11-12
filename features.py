import numpy as np
import pandas as pd

RADII = (50, 100, 250)


def load(train_path, test_path):
    tr = pd.read_csv(train_path, parse_dates=["date"])
    te = pd.read_csv(test_path, parse_dates=["date"])
    tr["is_test"] = 0
    te["is_test"] = 1
    te["rainfall_next_day_mm"] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df["id"] = df["id"].astype("float64")
    return df


def _haversine_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[None, :] - la[:, None]
    dlo = lo[None, :] - lo[:, None]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_features(df, use_neighbors=True, use_upwind=True):
    """Return (feature DataFrame, list of feature columns).
    Only same-day information is used (no lags, no next-day rows), because the
    test file contains neither previous- nor next-day rows."""
    df = df.copy()
    df["year"] = df.date.dt.year
    df["mon"] = df.date.dt.month
    doy = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_monsoon"] = df.mon.isin([6, 7, 8, 9]).astype(int)
    df["trange"] = df.max_temp - df.min_temp
    df["log_rain_today"] = np.log1p(df.rainfall_today_mm)

    # site id: name + coordinates (8 station names have two coordinate variants)
    key = (df.station_name + "|" + df.latitude.round(3).astype(str) + "|" +
           df.longitude.round(3).astype(str))
    df["site"] = key.astype("category").cat.codes
    df["state_code"] = df.state.astype("category").cat.codes

    # station-month climatology of the *input* variables (no target used)
    g = df.groupby(["site", "mon"])
    for c in ["air_pressure", "avg_temp", "rainfall_today_mm", "wind_speed", "trange"]:
        df[c + "_clim"] = g[c].transform("mean")
    df["p_anom"] = df.air_pressure - df.air_pressure_clim
    df["t_anom"] = df.avg_temp - df.avg_temp_clim
    df["w_anom"] = df.wind_speed - df.wind_speed_clim
    df["rain_anom"] = df.rainfall_today_mm - df.rainfall_today_mm_clim
    df["trange_anom"] = df.trange - df.trange_clim

    base = ["latitude", "longitude", "elevation", "avg_temp", "min_temp", "max_temp",
            "wind_speed", "air_pressure", "rainfall_today_mm", "log_rain_today", "mon",
            "doy_sin", "doy_cos", "is_monsoon", "trange", "state_code",
            "air_pressure_clim", "rainfall_today_mm_clim", "p_anom", "t_anom", "w_anom",
            "rain_anom", "trange_anom"]
    feats = list(base)
    if not use_neighbors:
        return df, feats

    # ---- neighbour features, computed within each calendar date -------------
    sites = df.drop_duplicates("site").sort_values("site")
    lat = sites.latitude.values
    lon = sites.longitude.values
    D = _haversine_matrix(lat, lon).astype(np.float32)
    latm = np.radians(lat.mean())
    X = ((lon - lon.mean()) * 111.32 * np.cos(latm)).astype(np.float32)
    Y = ((lat - lat.mean()) * 110.57).astype(np.float32)
    DX = X[None, :] - X[:, None]
    DY = Y[None, :] - Y[:, None]

    n = len(df)
    out = {}

    def alloc(name):
        out[name] = np.full(n, np.nan, dtype=np.float32)
        return out[name]

    arrs = {}
    for r in RADII:
        for k in ["cnt", "rain_mean", "rain_max", "wet_frac", "p_diff", "t_diff"]:
            arrs[f"nb{r}_{k}"] = alloc(f"nb{r}_{k}")
    for k in ["N", "E", "S", "W"]:
        arrs[f"sec_{k}_rain"] = alloc(f"sec_{k}_rain")
    if use_upwind:
        for k in ["gx", "gy", "gmag", "up_cnt100", "up_rain100", "up_rain250",
                  "up_rainmax250", "up_pdiff250", "up_adv"]:
            arrs[f"up_{k}"] = alloc(f"up_{k}")

    site = df.site.values
    rain = df.rainfall_today_mm.values.astype(np.float32)
    pan = df.p_anom.values.astype(np.float32)
    tan = df.t_anom.values.astype(np.float32)
    wsp = df.wind_speed.values.astype(np.float32)

    for _, idx in df.groupby("date").indices.items():
        s = site[idx]
        Dm = D[np.ix_(s, s)]
        DXm = DX[np.ix_(s, s)]
        DYm = DY[np.ix_(s, s)]
        R, P, T = rain[idx], pan[idx], tan[idx]
        eye = ~np.eye(len(idx), dtype=bool)
        for r in RADII:
            M = (Dm <= r) & eye
            cnt = M.sum(1)
            ok = cnt > 0
            c = np.maximum(cnt, 1)
            Mf = M.astype(np.float32)
            arrs[f"nb{r}_cnt"][idx] = cnt
            arrs[f"nb{r}_rain_mean"][idx] = np.where(ok, (Mf @ R) / c, np.nan)
            arrs[f"nb{r}_rain_max"][idx] = np.where(ok, np.where(M, R[None, :], -1).max(1), np.nan)
            arrs[f"nb{r}_wet_frac"][idx] = np.where(ok, (Mf @ (R >= 1).astype(np.float32)) / c, np.nan)
            arrs[f"nb{r}_p_diff"][idx] = np.where(ok, (Mf @ P) / c - P, np.nan)
            arrs[f"nb{r}_t_diff"][idx] = np.where(ok, (Mf @ T) / c - T, np.nan)
        M = (Dm <= 250) & eye
        Mf = M.astype(np.float32)
        # compass sectors (rain mean per sector within 250 km)
        sectors = {"N": DYm > np.abs(DXm), "S": -DYm > np.abs(DXm),
                   "E": DXm > np.abs(DYm), "W": -DXm > np.abs(DYm)}
        for k, sm in sectors.items():
            Ms = M & sm
            cs = Ms.sum(1)
            arrs[f"sec_{k}_rain"][idx] = np.where(cs > 0, (Ms.astype(np.float32) @ R) / np.maximum(cs, 1), np.nan)
        if use_upwind:
            # Wind direction is NOT in the data. Proxy: geostrophic-style flow from the
            # local pressure-anomaly gradient (wind keeps low pressure on its left in the NH).
            dp = P[None, :] - P[:, None]
            gx = (Mf * dp * DXm).sum(1) / ((Mf * DXm ** 2).sum(1) + 1e-3)
            gy = (Mf * dp * DYm).sum(1) / ((Mf * DYm ** 2).sum(1) + 1e-3)
            gmag = np.sqrt(gx ** 2 + gy ** 2)
            has = (M.sum(1) > 0) & (gmag > 1e-6)
            ux = np.where(has, gy / np.maximum(gmag, 1e-9), 0)   # unit vector pointing UPWIND
            uy = np.where(has, -gx / np.maximum(gmag, 1e-9), 0)
            dist = np.maximum(Dm, 1e-3)
            cosang = (DXm * ux[:, None] + DYm * uy[:, None]) / dist
            U = (cosang > 0.7071) & eye          # neighbours inside a 90-degree upwind cone
            U250, U100 = U & (Dm <= 250), U & (Dm <= 100)
            c250, c100 = U250.sum(1), U100.sum(1)
            arrs["up_gx"][idx] = np.where(has, gx, np.nan)
            arrs["up_gy"][idx] = np.where(has, gy, np.nan)
            arrs["up_gmag"][idx] = np.where(has, gmag, np.nan)
            arrs["up_up_cnt100"][idx] = np.where(has, c100, np.nan)
            arrs["up_up_rain100"][idx] = np.where(has & (c100 > 0), (U100.astype(np.float32) @ R) / np.maximum(c100, 1), np.nan)
            ur250 = np.where(has & (c250 > 0), (U250.astype(np.float32) @ R) / np.maximum(c250, 1), np.nan)
            arrs["up_up_rain250"][idx] = ur250
            arrs["up_up_rainmax250"][idx] = np.where(has & (c250 > 0), np.where(U250, R[None, :], -1).max(1), np.nan)
            arrs["up_up_pdiff250"][idx] = np.where(has & (c250 > 0), (U250.astype(np.float32) @ P) / np.maximum(c250, 1) - P, np.nan)
            arrs["up_up_adv"][idx] = ur250 * wsp[idx]

    for k, v in arrs.items():
        df[k] = v
    feats = base + [k for k in arrs if k.startswith("nb")] + [k for k in arrs if k.startswith("sec_")]
    if use_upwind:
        feats += [k for k in arrs if k.startswith("up_")]
    return df, feats
