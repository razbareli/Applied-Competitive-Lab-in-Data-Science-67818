import sqlite3

import pandas as pd
import requests
import json
import numpy as np
import asyncio
from tqdm.asyncio import tqdm
import datetime
import geopandas as gpd
import pyproj
from dateutil import rrule
import matplotlib.pyplot as plt
import os
import plotly.express as plx

Token = "GHpULthJRWoPphwLnbtqbgAVIUfityMx"
VALID_RESPONSE_STATUS = 200


async def get_info_from_rcc(long, lat, year, month=1, day=1):
    return await asyncio.get_event_loop().run_in_executor(None, requests.get,
                                                          f"http://data.rcc-acis.org/MultiStnData?bbox="
                                                          f"{long - 0.25},{lat - 0.25}"
                                                          f",{long + 0.25},{lat + 0.25}"
                                                          f"&date={year}-{month}-{day}&elems=1,2,4,7,43")


def get_result(ls):
    if not ls:
        return np.nan
    else:
        return np.mean(ls)


async def get_weather_info(row, sem, year, day, month):
    """
    A function that gets a pandas series with FIRE_YEAR, MONTH, DAY, LONGITUDE, LATITUDE columns
    and returns the minimal temperature, maximal temperature, Precipitation, and average temperature
    in the area denoted by LONGITUDE LATITUDE in the day before the dat FIRE_YEAR, MONTH, DAY.

    :param row: A pandas dataframe with FIRE_YEAR, MONTH, DAY, LONGITUDE, LATITUDE columns
    :param sem: A semaphore to limit the threading of the function to avoid getting banned from the server, should be initialized with 20.
    :return: minimal temp, maximal temp, Precipitation, average temp.
    """
    while True:
        async with sem:
            try:
                r = await get_info_from_rcc(row['geometry'].x.iloc[0], row['geometry'].y.iloc[0], year, month, day)
            except requests.exceptions.ConnectionError:
                await asyncio.sleep(60)
                continue

        if r.status_code == VALID_RESPONSE_STATUS:
            try:
                d = json.loads(r.text)
                break
            except json.decoder.JSONDecodeError:
                continue

        await asyncio.sleep(10)

    stations = [item for item in d['data']]

    max_temp = [float(item["data"][0]) for item in stations if is_float(item["data"][0])]
    min_temp = [float(item["data"][1]) for item in stations if is_float(item["data"][1])]
    prcp = [float(item["data"][2]) for item in stations if is_float(item["data"][2])]
    Pan_evaporation = [float(item["data"][3]) for item in stations if is_float(item["data"][3])]
    avg_temp = [float(item["data"][4]) for item in stations if is_float(item["data"][4])]

    max_temp = get_result(max_temp)
    min_temp = get_result(min_temp)
    prcp = get_result(prcp)
    Pan_evaporation = get_result(Pan_evaporation)
    avg_temp = get_result(avg_temp)

    return max_temp, min_temp, prcp, Pan_evaporation, avg_temp


def is_float(str_num):
    try:
        float(str_num)
        return True
    except ValueError:
        return False


async def get_temps_for_df_prer_year(df: gpd.GeoDataFrame, year: int):
    """
    A function that gets a DataFrame and returns min_temp, max_temp, precipitation and average temp
    for each row.
    :param year:
    :param df:
    :return:
    """
    sem = asyncio.Semaphore(20)

    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)
    days = [d1 + datetime.timedelta(days=x) for x in range(0, (d2 - d1).days + 1, 3)]

    for day in days:
        if not os.path.exists(f"temps_dfs/{day.year}/{day.month}/{day.day}.csv"):
            results = np.array(await tqdm.gather(*(get_weather_info(df.iloc[[i]], sem, day.year, day.day, day.month)
                                                   for i in range(len(df)))))

            df["max_temp"] = results[:, 0]
            df["min_temp"] = results[:, 1]
            df["prcp"] = results[:, 2]
            df["Pan_evaporation"] = results[:, 3]
            df["avg_temp"] = results[:, 4]

            if not os.path.exists(f"temps_dfs"):
                os.mkdir(f"temps_dfs")
            if not os.path.exists(f"temps_dfs/{day.year}"):
                os.mkdir(f"temps_dfs/{day.year}")
            if not os.path.exists(f"temps_dfs/{day.year}/{day.month}"):
                os.mkdir(f"temps_dfs/{day.year}/{day.month}")

            df.to_csv(f"temps_dfs/{day.year}/{day.month}/{day.day}.csv", index=False)

    return df


async def get_temps():
    df = pd.read_csv("all_us_points.csv")
    df = gpd.GeoDataFrame(df.loc[:, [c for c in df.columns if c != "geometry"]],
                          geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs="epsg:4326")

    for i in range(1992, 2016):  # todo: change to single year before running
        print(f"starting year: {i}")
        df = await get_temps_for_df_prer_year(df, i)


async def main():
    await get_temps()


def create_pop_csv():
    df = gpd.read_file("data_populations.csv")
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    df = df.drop(df[df.x > -20].index)
    df = df.drop(df[df.y < -20].index)
    df = df.drop(["x", "y"], axis=1)
    df.to_csv("data_populations_usa.csv", index=False)


def to_date(x):
    date = datetime.datetime(x['FIRE_YEAR'], 1, 1) + datetime.timedelta(x['DISCOVERY_DOY'] - 1)
    return date


def date_to_check(row):
    day_to_check = row['DISCOVERY_DOY'] - (1 + (row['DISCOVERY_DOY'] - 1) % 3)
    date = datetime.datetime(row['YEAR'], 1, 1) + datetime.timedelta(int(day_to_check))

    if date.year < 1992:
        return 1992, 1, 1
    else:
        return date.year, date.month, date.day


def temps_func(df):
    gdf = pd.read_csv("datasets/temps_dfs/temps_area_codes.csv")
    gdf = gpd.GeoDataFrame(gdf.loc[:, [c for c in gdf.columns if c != "geometry"]],
                           geometry=gpd.GeoSeries.from_wkt(gdf["geometry"]))

    gdf.crs = "EPSG:4326"

    gdf = gdf.to_crs("EPSG:3857")
    df.crs = gdf.crs

    df = gpd.sjoin_nearest(df, gdf, how="left")

    df = df.drop(["index_right"], axis=1)

    df["day_remove"] = df["DAY"]
    df["month_remove"] = df["MONTH"]
    df["year_remove"] = df["YEAR"]

    dates = np.array(list(df.apply(date_to_check, axis=1)))

    df["DAY"] = dates[:, 2]
    df["MONTH"] = dates[:, 1]
    df["YEAR"] = dates[:, 0]

    gdf = pd.read_csv(f"datasets/temps_dfs/temps_area_code_dates.csv")

    gdf["YEAR"] = gdf["YEAR"].astype("int32")
    df["YEAR"] = df["YEAR"].astype("int32")
    gdf["MONTH"] = gdf["MONTH"].astype("int32")
    df["MONTH"] = df["MONTH"].astype("int32")
    gdf["DAY"] = gdf["DAY"].astype("int32")
    df["DAY"] = df["DAY"].astype("int32")

    df = pd.merge(df, gdf, on=["DAY", "MONTH", "YEAR", "area_code"], how="left")

    df["DAY"] = df["day_remove"]
    df["MONTH"] = df["month_remove"]
    df["YEAR"] = df["year_remove"]
    df = df.drop(["area_code", "day_remove", "month_remove", "year_remove"], axis=1)
    df = df.drop_duplicates(subset=["FOD_ID"])

    return df


if __name__ == '__main__':
    asyncio.run(main())
