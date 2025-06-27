from bs4 import BeautifulSoup
import sys
import traceback
import requests
import time
import csv
import json
import time
import os
import asyncio
from pyppeteer import launch
from pyppeteer_stealth import stealth


# STEPS
# 1. Put data from every time split into a new csv file.
# 2. Save csv files in folder. 



async def scrape_data():

    # Plan to scrape data from the website barttorvik.com for college basketball players.
    # 1. Load website
    # 2. Press the "Load More" button to load all player data.
    # 3. Extract player data from the page.

    date_list = ["1101", "1114", "1201", "1214", "0101", "0114", "0201", "0214", "0301", "0314", "0401", "0414", "0501", "0514", "0601"]

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )

    browser = await launch({
           'headless': True,
            'args': ['--no-sandbox', '--disable-setuid-sandbox', '--disable-infobars', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled'],
           'ignoreHTTPSErrors': True
        })

    page = await browser.newPage()
    await page.setUserAgent(user_agent)
    await stealth(page)  # Makes the page more human-like to websites


    for i in range(2008, 2026, 1): # Get year range from 2008 to 2025
        print(f"Scraping data for year: {i}")

        for k in range(0, len(date_list)-1): # Loop through each data range in date_list

            time_split_list = []

            date_list_month = int(date_list[k][0])

            url_year = i+1 if date_list_month == 1 else i

            full_start_date = f"{i}{date_list[k]}"
            full_end_date = f"{i}{date_list[k+1]}"

            # first date = https://barttorvik.com/playerstat.php?link=y&minGP=1&year=2009&start=20081101&end=20081114
            url = f"https://barttorvik.com/playerstat.php?link=y&minGP=1&year={url_year}&start={full_start_date}&end={full_end_date}"



            await page.goto(url)
            await page.waitForSelector('table')
            
            await page.querySelector('[style="white-space:nowrap;margin:auto;table-layout:fixed;"]')

            tables = await page.querySelectorAll('table')

            counter = 0
            while len(tables) < 2 and counter < 20.0: # If the table does not load in 20 seconds, skip this date range
                await asyncio.sleep(0.25)  # Wait for the page to load more tables
                tables = await page.querySelectorAll('table')
                counter += 0.25
                continue

            more_button = await page.querySelector('td[id="expand"]')

            print(f"more_button: {more_button}")

            for i in range(40):
                if more_button is not None:
                    await more_button.click()
                    await asyncio.sleep(0.1) 


            tables = await page.querySelectorAll('table')


            hidden_table = tables[1]

            table_text = await page.evaluate('(el) => el.innerText', hidden_table)
            table_list = table_text.split('\n')

            time_split_list = []


            print(f"Found {len(tables)} tables")
            print(f"table: {hidden_table}")
            print(f"table text: {type(table_text)}")

            # print(f"table list: {table_list}") # table_text is a string

            # Get inner text of the table
            '''table_text = await page.evaluate('(el) => el.innerText', table)
            print(table_text)'''

            # FIND WAY TO DELETE LAST TWO ROWS OF THE TABLE IF THE ARE THE BUTTON ROWS


            for row in table_list:

                if len(row) < 6:  # Skip rows that are too short
                    continue
                
                row_data = row.split('\t')  # Split the row by tab character

                if row_data[0] == 'SHOW 100 MORE' or row_data[0] == 'SHOW 50 MORE':  # Skip the "SHOW MORE" rows
                    break

                if row_data[-1] == '3P': # If this row is the title row, skip it
                    continue

                # print(f"new row_data: {row_data}")

                player_height = row_data[1]
                player_inches = float(player_height[2:])
                scaled_player_inches = player_inches / 12.0
                final_player_height = f"{float(player_height[0])+scaled_player_inches}"

                # print(f"final_player_height: {final_player_height}")

                edited_row_data = [row_data[2]] + row_data[5:-5] + row_data[-4:-2] + [row_data[-1]] + [final_player_height]  # Create a new list with the desired columns
                time_split_list.append(edited_row_data)

            for row in time_split_list:
                print(f"new row: {row}")


            await browser.close()


            break

            
            # Find and press the "Load More" button until all data is loaded
        break
    

    return


def main():
    # Steps:
    # 1. Find and import all data from college basketball players. Need to make sure each piece of data is a monthly, bimonthly, weekly, etc spread.
    # 2. Clean and format data into structure needed to train a model.
    # 3. Train a model to predict a data point that accuratly represents a player's performance.
    # 4. Evaluate the model's performance.

    # 5. Use the model to predict future performance.
    # 6. Visualize the results and predictions.
    # 7. Save the model for future use.


    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already running (e.g., IPython/Jupyter), use ensure_future
            task = asyncio.ensure_future(scrape_data())
            loop.run_until_complete(asyncio.gather(task))
        else:
            loop.run_until_complete(scrape_data())
        return
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
    
        return
    
    return

    # Next Steps:
    # 1. Save all data from a single player, from multiple time splits, into one row of a CSV file. (repeat for each player)
    # 2. Make function/file that finds all players with the right amount of data (e.g. 12 consecute time splits) and saves them 
    #    into a seperate CSV file to be put into a model. (It will have to be decided what type of data is acceptable for the model, 
    #     e.g. only allowing splits over 1 season, 2 seasons, whether time splits covering different numbers of seasons can be included
    #     together, etc.)
    # 3. Clean the data. Make sure you scale the data right, like what the youtube video said to do. 
    # 4. Make a model that can predict a player's performance based on the data.
    
    # 5. Evaluate the model's performance.
    # 6. Use the model to predict future performance.
    # 7. Visualize the results and predictions.

    # 1. Save the scraped data to a CSV file.

    '''with open('bball_data.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Player', 'Height', 'Weight', 'Position', 'Team', 'Final Height'])
        for player_data in time_split_list:
            csvwriter.writerow(player_data)'''


if __name__ == "__main__":
    main()