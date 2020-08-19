# option 1 Weather Underground
# scrape https://www.wunderground.com/history/monthly/lb/beirut/OLBA/date/2014-12
# paid api, waiting to see if they will provide for free


# option 2
# scrape https://www.timeanddate.com/weather/results.html?query=lebanon
# data points in most mohafazas

from bs4 import BeautifulSoup
import pandas as pd
import selenium
from selenium import webdriver
import time


def generate_url(centre, year, month):
    return f"https://www.wunderground.com/history/monthly/lb/beirut/{centre}/date/{year}-{month}"


def process_weather_text(table, year, month, station):
    """
    Takes the weather data from one month and processes as a dataframe
    :param table: the full weather table as a selenium web element
    :return: df a data frame of the weather for the given month
    """
    df = pd.DataFrame(columns=["Mohafaza", "Date", "MaxTemp", "MinTemp", "AverageTemp", "MaxHumidity", "MinHumidity", "MaxWindSpeed", "MinWindSpeed", "Precipitation"])
    html = BeautifulSoup(table.get_attribute("innerHTML"), 'html.parser')
    columns = html.find_all('table')
    num_days = len(columns[0].find_all("td")) - 1

    # Set date and mohafaza
    df.Date = [f"{year}-{month}-{i}" for i in range(1, num_days + 1)]
    df.Date = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.Mohafaza = [station_to_mohafaza[station]] * num_days

    # Extract data from table, skipping headers, stripping whitespace and converting to numeric
    df.MaxTemp = [int(td.text) for td in columns[1].find_all("td")[3::3]]
    df.AverageTemp = [int(td.text) for td in columns[1].find_all("td")[4::3]]
    df.MinTemp = [int(td.text) for td in columns[1].find_all("td")[5::3]]

    df.MaxHumidity = [int(td.text) for td in columns[3].find_all("td")[3::3]]
    df.MinHumidity = [int(td.text) for td in columns[3].find_all("td")[5::3]]

    df.MaxWindSpeed = [int(td.text) for td in columns[4].find_all("td")[3::3]]
    df.MinWindSpeed = [int(td.text) for td in columns[4].find_all("td")[5::3]]

    df.Precipitation = [float(td.text) for td in columns[6].find_all("td")[4::3]]

    # Convert from Farenheit to Degrees Celcius
    for col in ["MaxTemp", "MinTemp", "AverageTemp"]:
        df[col] = (df[col] - 32) * 5 / 9

    # Convert from mph to km/h
    for col in ["MaxWindSpeed", "MinWindSpeed"]:
        df[col] = df[col] * 1.609

    # Convert from inches to mm
    df["Precipitation"] = df["Precipitation"] * 25.4

    return df




station_to_mohafaza = {"OLBA": "Beirut",
                       "IMOUNTLE8": "Mount Lebanon",
                       "ICHOUFMO2": "South Lebanon",
                       "IWESTERN748": "Nabatiye",
                       "IBIQAAER2": "Bekaa",
                       "IBAALBEK3": "Baalbek",
                       "I5340": "North Lebanon"}

stations = ["OLBA", "IJLEJJOU2", "IBAABDA7", "IBEIRUTB3", "IMOUNTLE13", "I5325", "IMATNDIS11", "IBEIRUTB2", "IMATNDIS8",
            "IMOUNTLE9", "IBAABDA11", "IMOUNTLE8", "IKESERWA2", "IWESTERN748", "IBIQAAER2", "IBAALBEK3", "INABATIE7",
            "I5427", "IALKOURA2", "INORTHBC2", "I5340", "IMANARA2", "IQIRYATS3", "INORTHDA2", "IYIFTAH3", "INORTHER146",
            "IHILA2", "IMAALOTT3", "IISRAELS3", "IEINYAAQ2", "INORTHER127", "INORTHER145", "I1341", "IAKKA3", "INORTHQO2"]



weatherDF = pd.DataFrame(columns=["Mohafaza", "Date", "MaxTemp", "MinTemp", "AverageTemp", "MaxHumidity", "MinHumidity", "MaxWindSpeed", "MinWindSpeed", "Precipitation"])
browser = webdriver.Chrome()
for weathercentre in stations:
    for year in [2014, 2015, 2016]:
        for month in range(1,13):
            url = generate_url(weathercentre, year, month)
            browser.get(url)
            table = None
            start = time.time()
            while time.time() - start < 10:
                try:
                    table = browser.find_element_by_class_name("days")
                except selenium.common.exceptions.NoSuchElementException:
                    pass
                else:
                    weatherDF = weatherDF.append(process_weather_text(table, year, month, weathercentre),
                                                 ignore_index=True)
                    break



browser.close()
weatherDF = weatherDF.round(1)
print(weatherDF.size)

weatherDF.to_csv('output/weather/Weather.csv', index=False)