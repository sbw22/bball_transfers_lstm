from bs4 import BeautifulSoup
import sys
import traceback
import requests
import time
import csv
import json
import os
import asyncio
import nest_asyncio
from pyppeteer import launch
from pyppeteer_stealth import stealth
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException, NoSuchElementException




# STEPS
# 1. Put data from every time split into a new csv file.
# 2. Save csv files in folder. 



def scrape_data():

    player_dict = {}  # Dictionary to hold player data

    # Plan to scrape data from the website barttorvik.com for college basketball players.
    # 1. Load website
    # 2. Press the "Load More" button to load all player data.
    # 3. Extract player data from the page.

    # date_list = ["1101", "1114", "1201", "1214", "0101", "0114", "0201", "0214", "0301", "0314", "0401", "0414", "0501", "0514", "0601"]
    date_list = ["1101", "1114", "1201", "1214", "0101", "0114", "0201", "0214", "0301", "0314", "0401"]


    chrome_options = Options()
    # chrome_options.add_argument("--headless=new")  # Use new headless mode (from Chrome 109+)
    #chrome_options.add_argument("--disable-gpu")
    #chrome_options.add_argument("--no-sandbox")
    #chrome_options.add_argument("--disable-dev-shm-usage")


    path = '/Users/spencerweishaar/Downloads/chromedriver-mac-arm64/chromedriver'
    service = Service(executable_path=path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    



    for i in range(2024, 2025, 1): # Get year range (might want to expand range to 2008 when I have mroe time to scrape data)
        if i == 2019:  # Skip the years 2019 and 2020 due to COVID-19 season
            print(f"Skipping year {i} due to COVID-19 season.")
            continue
        print(f"Scraping data for year: {i}")

        for k in range(0, len(date_list)-1): # Loop through each data range in date_list

            if k % 5 == 0 and k != 0:
                driver.quit()
                driver = webdriver.Chrome(service=service, options=chrome_options)  # Restart the driver every 5 iterations to avoid memory issues
            driver.delete_all_cookies()

            print(f"Scraping data for date range: {date_list[k]} to {date_list[k+1]}")

            date_list_month = int(date_list[k][:2])

            url_year = i+1 # if date_list_month < 10 else i

            year_1 = i if date_list[k][0] == "1" else i+1  
            year_2 = url_year if date_list[k][:2] == "12" and date_list[k+1][:2] == "01" else year_1  # If the date range is from December to January, increment the year by 1
            time_split = f"{year_1}{date_list[k]}-{year_2}{date_list[k+1]}"

            full_start_date = f"{year_1}{date_list[k]}"
            full_end_date = f"{year_2}{date_list[k+1]}"

            if int(full_start_date) > 20250901:  # If the start date is after September 1, 2025, skip it
                print(f"Skipping date range {full_start_date} to {full_end_date} as it is after September 1, 2025.")
                continue
            

            # first date = https://barttorvik.com/playerstat.php?link=y&minGP=1&year=2009&start=20081101&end=20081114
            url = f"https://barttorvik.com/playerstat.php?link=y&minGP=1&year={url_year}&start={full_start_date}&end={full_end_date}"

            
            driver.get(url)

            time.sleep(2)  # Wait for the page to load
        
            table_counter = 3
            while True:
                table_counter += 1
                # Prevent infinite loop if the table never loads
                if table_counter > 7:
                    table_counter = 7

                driver.refresh()  # Refresh the page to ensure all elements are loaded
                time.sleep(2*table_counter)  # Wait for the page to load

                tables = driver.find_elements(By.CSS_SELECTOR, 'table')
                if len(tables) >= 2:  # Check if the table is loaded
                    break

            

            tables = driver.find_elements(By.CSS_SELECTOR, 'table')

            if len(tables) < 2:
                print("Failed to load tables.")
                continue

            print(f"Tables found: {len(tables)}")

            more_button = driver.find_element(By.ID, 'expand')
            print(f"More button found: {more_button}")


            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'expand'))
            )

            for j in range(40):
                if load_more_button is not None:
                    load_more_button.click()
                    time.sleep(0.1)
            

            
            tables = driver.find_elements(By.CSS_SELECTOR, 'table')

            hidden_table = tables[1]

            try:
                table_text = hidden_table.text
            except Exception as e:
                print(f"Failed to get text from hidden_table: {e}")
                continue

            table_list = table_text.split('\n')

            
            
            for row in table_list:  # Iterate through each row in the table

                # print(f"Row: {row}")

                row_data = row.split(' ')
                
                if len(row) < 6:  # Skip rows that are too short
                    continue
                if row_data[0] == 'SHOW':  # Skip the "SHOW MORE" rows
                    break
                if row_data[-1] == '3P': # If this row is the title row, skip it
                    continue

                # print(f"row_data: {row_data}")


                # print(f"new row_data: {row_data}")

                try:
                    player_height = row_data[0]
                    player_inches = float(player_height[2:])
                    scaled_player_inches = player_inches / 12.0
                    final_player_height = f"{float(player_height[0])+scaled_player_inches}"
                except ValueError as e:
                    print(f"Error converting player height to float: {e}. Skipping this player.")
                    continue

                # print(f"final_player_height: {final_player_height}")

                player_name = f"{row_data[1]} {row_data[2]}"  # Combine first and last name

                player_identifier = f"{row_data[3:-19]}" # Includes the player's team, conference, and other identifiers               
            
                edited_row_data = [[player_name]] + row_data[-19:-5] + row_data[-4:-2] + [row_data[-1]] + [final_player_height]  # Create a new list with the desired columns

                pure_row_data = edited_row_data[1:]

                # print(f"row_data: {row_data}")
                try:
                    pure_row_data = [float(stat) for stat in pure_row_data[:-1]]  # Convert all stats to floats
                except ValueError as e:
                    print(f"Error converting row_data to floats: {e}. skipping this player.")
                    print(f"row_data: {row_data}")
                    return
                    continue

                if player_name in player_dict and player_identifier in player_dict[player_name]:  # Check if player already exists in the dictionary and has an identifier unique to them
                    player_dict[player_name][0].append([pure_row_data, time_split])
                else:
                    player_dict[player_name] = [[[pure_row_data, time_split]], player_identifier]




            
                # print(f"new row_data: {row_data}")



            

            # time.sleep(200)

    driver.quit()  # Close the browser after scraping all data
    print(f"Total players scraped: {len(player_dict)}")

    
    print("Returning player_dict from scrape_data")
    return player_dict


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
        '''loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already running (e.g., IPython/Jupyter), use ensure_future
            task = asyncio.ensure_future(scrape_data())
            player_dict_result = loop.run_until_complete(asyncio.gather(task))
            print(f"back")
            player_dict = player_dict_result[0]
        else:
            player_dict = loop.run_until_complete(scrape_data())
            print(f"back2")'''


        player_dict = scrape_data()

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
    

    
    # Now we have a dictionary of players and their data, we can save it to a CSV file or process it further.
    # Save the player_dict to a CSV file
    with open('selenium_data/biweekly/player_data_2024-25_biweekly.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Player', 'Stats'])
        for player_data in player_dict:
            csvwriter.writerow([player_data, player_dict[player_data]])
    csvfile.close()
    print("Player data saved to selenium_data/biweekly/player_data_2024-25_biweekly.csv")

    
    return

    # FINISH UP MAKING ALL THE NUMBERS FLOATS, AND REMOVE THE BREAKS SO THAT IT SCRAPES ALL THE DATA

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




if __name__ == "__main__":
    main()