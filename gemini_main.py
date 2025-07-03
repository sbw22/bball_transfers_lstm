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

async def scrape_data():
    player_dict = {}  # Dictionary to hold player data

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

            print(f"Scraping data for date range: {date_list[k]} to {date_list[k+1]}")

            time_split_list = []

            # Correcting the year logic for URL construction
            # If the start date month is November or December (11, 12), the year for the URL is 'i'.
            # If the start date month is January through October (01-10), the year for the URL is 'i+1'.
            # This handles the college basketball season spanning two calendar years.
            start_month = int(date_list[k][:2])
            url_year_for_season = i if start_month >= 11 else i + 1

            full_start_date = f"{i}{date_list[k]}"
            # The end date also needs careful year consideration if it crosses calendar years.
            # If the start month is Dec (12) and end month is Jan (01), then the end year is 'i+1'.
            # Otherwise, the end year is the same as the start year 'i'.
            end_month = int(date_list[k+1][:2])
            year_for_end_date = i + 1 if (start_month == 12 and end_month == 1) else i
            full_end_date = f"{year_for_end_date}{date_list[k+1]}"


            # Determine time_split string for player_dict
            # The time_split string should reflect the actual season year for the start and end of the split.
            # This needs to be consistent with how the full_start_date and full_end_date are constructed.
            time_split_start_year = i
            if start_month < 11: # If starting in Jan-Oct, it's the next calendar year's season
                time_split_start_year = i + 1

            time_split_end_year = i
            if end_month < 11: # If ending in Jan-Oct, it's the next calendar year's season
                time_split_end_year = i + 1
            # Special case for Dec to Jan, where end year increments
            if start_month == 12 and end_month == 1:
                time_split_end_year = i + 1


            time_split = f"{time_split_start_year}{date_list[k]}-{time_split_end_year}{date_list[k+1]}"


            url = f"https://barttorvik.com/playerstat.php?link=y&year={url_year_for_season}&start={full_start_date}&end={full_end_date}"

            try:
                await page.goto(url, timeout=30000) # Increased timeout
            except Exception as e:
                print(f"Failed to load URL: {url} — {e}")
                continue

            # Wait for the main content to be visible, specifically the container for the tables.
            # This is often more reliable than just waiting for 'table'.
            try:
                await page.waitForSelector('div[style="white-space:nowrap;margin:auto;table-layout:fixed;"]', timeout=10000)
            except Exception as e:
                print(f"Main table container not found or loaded for {url}: {e}")
                continue

            # Give a small buffer time after the container is found, sometimes content renders slightly after.
            await asyncio.sleep(0.5)

            # Re-check for tables after the main container is present
            tables = await page.querySelectorAll('table')
            # Now, wait specifically for the second table to be present and visible
            try:
                # Wait for the second table to be present in the DOM.
                # A more specific selector for the second table might be even better if available.
                await page.waitForSelector('table:nth-of-type(2)', timeout=15000)
                tables = await page.querySelectorAll('table') # Re-fetch tables after waiting
                if len(tables) < 2:
                    print(f"Only {len(tables)} tables found after waiting for the second table for {url}. Skipping.")
                    continue # Skip this iteration if the second table still isn't there
            except Exception as e:
                print(f"Second table not found or loaded for {url}: {e}")
                continue


            # Check if tables[1] exists before trying to access it
            if len(tables) < 2:
                print(f"Insufficient tables found ({len(tables)}) for {url}. Expected 2. Skipping.")
                continue

            hidden_table = tables[1] # This is the player data table

            # Find the "Load More" button. Important to refetch inside the loop if it disappears or changes.
            more_button = await page.querySelector('td[id="expand"]')
            
            # Continuously click "Load More" until it's no longer visible
            # We need to ensure new rows are loaded before checking for the button again.
            while more_button:
                try:
                    await more_button.click()
                    # Wait for new content to potentially load after clicking
                    # A small delay is crucial here. 0.25-0.5 seconds is usually good.
                    await asyncio.sleep(0.5) 
                    more_button = await page.querySelector('td[id="expand"]') # Re-fetch the button
                except Exception as e:
                    print(f"Error clicking 'Load More' or button disappeared: {e}")
                    more_button = None # Assume button is gone if we can't click it
                    break # Exit the loop

            # After all "Load More" clicks, get the final innerText of the target table.
            try:
                table_text = await page.evaluate('(el) => el.innerText', hidden_table)
            except Exception as e:
                print(f"Failed to evaluate hidden_table after load more: {e}")
                continue

            table_list = table_text.split('\n')
            
            # Debugging outputs
            print(f"Found {len(tables)} tables after all 'Load More' clicks.")
            # print(f"Raw table text length: {len(table_text)} characters") # Useful for large tables
            # print(f"Number of lines in table_list: {len(table_list)}")

            # Process the table_list
            current_time_split_players_data = [] # To store data for this specific time split

            for row in table_list:  # Iterate through each row in the table
                row = row.strip() # Remove leading/trailing whitespace
                if not row: # Skip empty rows
                    continue
                
                row_data = row.split('\t')  # Split the row by tab character

                # Refined conditions to skip non-data rows
                if len(row_data) < 6: # Too short to be a player row
                    continue
                if row_data[0] in ['SHOW 100 MORE', 'SHOW 50 MORE', 'Player']: # Skip button/header rows
                    continue
                if row_data[-1] == '3P' and row_data[0] == 'Rk': # This is likely a header row
                    continue

                # Ensure row_data has enough elements before accessing indices
                if len(row_data) < 10: # Minimum expected columns for player data
                    print(f"Skipping malformed row: {row_data}")
                    continue

                try:
                    player_height = row_data[1]
                    # Robust height parsing
                    if "'" in player_height and '"' in player_height:
                        feet, inches_str = player_height.split("'")
                        inches = float(inches_str.replace('"', ''))
                        final_player_height = float(feet) + (inches / 12.0)
                    else:
                        print(f"Warning: Unexpected height format '{player_height}' for row: {row_data}. Skipping height conversion.")
                        final_player_height = None # Or handle as an error

                    # Adjust column indices if the layout shifts or more columns are present
                    # Based on your original code, these indices are:
                    # Player Name: row_data[2]
                    # Stats: row_data[5:-5] + row_data[-4:-2] + [row_data[-1]]
                    # This assumes a consistent number of columns that are *not* player name, height, etc.
                    # It's safer to identify columns by their header if possible, or count carefully.
                    # For now, let's stick to your original indices and assume they are correct after cleaning.
                    
                    # Ensure you have enough elements for slicing before creating edited_row_data
                    if len(row_data) < 10: # Example minimum length based on your slicing. Adjust as needed.
                         print(f"Row too short for expected slices after cleaning: {row_data}. Skipping.")
                         continue

                    # The original slicing was: [row_data[2]] + row_data[5:-5] + row_data[-4:-2] + [row_data[-1]]
                    # Let's verify this against the actual table structure.
                    # Usually, column 0 is Rank, 1 is Height, 2 is Player Name, 3 is Team, 4 is Pos.
                    # Then stats begin.
                    # Your edited_row_data takes:
                    # row_data[2] -> Player Name
                    # row_data[5:-5] -> Some stats
                    # row_data[-4:-2] -> Other stats
                    # row_data[-1] -> Final stat (e.g., 3P%)
                    # The last element added is final_player_height

                    player_name = row_data[2] # Player Name
                    # Assuming stats are columns 5 onwards, and then specific last columns
                    # This needs careful verification against the live site structure.
                    # Let's take the columns from a specific range if the headers are reliable.
                    
                    # For robust parsing, it's better to explicitly map columns if possible
                    # Or, if the structure is fixed, ensure the slice is correct.
                    
                    # Re-evaluating your original slicing:
                    # [row_data[2]] (Player Name)
                    # + row_data[5:-5] (Stats in middle)
                    # + row_data[-4:-2] (More stats before the last few columns)
                    # + [row_data[-1]] (The very last stat)
                    
                    # This is highly dependent on the exact number of columns and their positions.
                    # A more resilient approach is to identify column headers and then extract data.
                    # However, if your current slicing works for the data, let's keep it but be aware.

                    # Let's assume row_data is structured like:
                    # [Rk, Height, Player Name, Team, Pos, Stat1, Stat2, ..., StatN-4, StatN-3, StatN-2, StatN-1, StatN]
                    # If edited_row_data is: [Player Name, Stat1, Stat2, ..., StatN-4, StatN-3, StatN-2, StatN-1, StatN, FinalHeight]
                    # This means row_data[3] (Team) and row_data[4] (Pos) are skipped.
                    # It also skips row_data[0] (Rk).
                    
                    # Pure stats would be everything from index 5 up to row_data.length - 5,
                    # then row_data.length - 4, row_data.length - 3, row_data.length - 1.

                    # Let's reconstruct pure_row_data to be more explicit if possible, or verify indices.
                    # Based on your previous working code, the indices seem to imply a structure.
                    # For safety, ensure row_data has enough elements to prevent IndexError.
                    
                    # Assuming row_data structure based on your original edited_row_data creation:
                    # row_data[0] = Rk
                    # row_data[1] = Height
                    # row_data[2] = Player
                    # row_data[3] = Team
                    # row_data[4] = Pos
                    # row_data[5] to row_data[len(row_data)-6] are the middle stats
                    # row_data[len(row_data)-5] and row_data[len(row_data)-2] are skipped
                    # row_data[len(row_data)-4] to row_data[len(row_data)-3] are part of the last stats
                    # row_data[len(row_data)-1] is the final stat (3P)

                    # Let's try to make the stat extraction more direct, assuming a fixed column order for stats:
                    # Example: if stats always start at index 5 and end at index -1 (excluding height/name/team/pos)
                    # This needs to be precisely matched with the website's table structure.
                    
                    # For now, let's use your original slicing and assume it's correct for the structure.
                    # Ensure minimum length check is robust enough for your slicing.
                    min_expected_len_for_slicing = 10 # Adjust this based on your actual table columns
                    if len(row_data) < min_expected_len_for_slicing:
                        print(f"Row data too short for slicing: {row_data}. Skipping.")
                        continue

                    stats_to_extract = []
                    # Add stats from index 5 to len(row_data) - 6
                    stats_to_extract.extend(row_data[5:len(row_data)-5])
                    # Add stats from index len(row_data)-4 to len(row_data)-3
                    stats_to_extract.extend(row_data[len(row_data)-4:len(row_data)-2])
                    # Add the very last stat
                    stats_to_extract.append(row_data[-1])

                    pure_row_data = [float(stat) for stat in stats_to_extract if stat.replace('.', '', 1).isdigit()] # Convert to float, safely

                    if final_player_height is not None:
                        pure_row_data.append(final_player_height) # Add height at the end

                    if player_name in player_dict:
                        player_dict[player_name].append([pure_row_data, time_split])
                    else:
                        player_dict[player_name] = [[pure_row_data, time_split]]

                    current_time_split_players_data.append(pure_row_data) # For local logging if needed

                except ValueError as ve:
                    print(f"ValueError converting data for row: {row_data} - {ve}. Skipping row.")
                    continue
                except IndexError as ie:
                    print(f"IndexError accessing row data: {row_data} - {ie}. Check slicing. Skipping row.")
                    continue
                except Exception as ex:
                    print(f"An unexpected error occurred processing row: {row_data} - {ex}. Skipping row.")
                    traceback.print_exc()
                    continue

            # print(f"Scraped {len(current_time_split_players_data)} players for time split: {time_split}")

    await browser.close()
    print("Returning player_dict from scrape_data")
    return player_dict


def main():
    try:
        nest_asyncio.apply()
        player_dict = asyncio.run(scrape_data())

    except Exception as e:
        print(f"Error occurred in main: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Debugging: Print a sample of the collected data
    print("\n--- Sample of Collected Player Data ---")
    sample_count = 0
    for player, data in player_dict.items():
        print(f"Player: {player}")
        for entry in data:
            print(f"  Stats: {entry[0]}, Time Split: {entry[1]}")
        sample_count += 1
        if sample_count >= 5: # Print data for first 5 players
            break
    print("---------------------------------------")

    # Save the player_dict to a CSV file
    output_directory = "scraped_player_data"
    os.makedirs(output_directory, exist_ok=True)
    output_filepath = os.path.join(output_directory, 'player_data.csv')

    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write a header that makes sense for your data structure
        # This header might need to be dynamically generated or hardcoded based on the exact stats.
        csvwriter.writerow(['Player', 'Time_Split', 'Stat_1', 'Stat_2', '...', 'Stat_N', 'Height_Feet_Decimal']) 
        
        for player_name, player_data_list in player_dict.items():
            for stats_entry, time_split in player_data_list:
                # stats_entry is a list of floats, time_split is a string
                row = [player_name, time_split] + stats_entry
                csvwriter.writerow(row)
    
    print(f"Player data saved to {output_filepath}")

if __name__ == "__main__":
    main()