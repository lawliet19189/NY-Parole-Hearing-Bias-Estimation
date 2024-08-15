# import requests
# from bs4 import BeautifulSoup
# import csv
# from datetime import datetime
# import time
# import string

# BASE_URL = "https://publicapps.doccs.ny.gov/ParoleBoardCalendar/"


# def get_interview_data(session, year, month, letter):
#     # Get the initial page to retrieve necessary cookies and hidden fields
#     initial_response = session.get(f"{BASE_URL}default")
#     initial_soup = BeautifulSoup(initial_response.content, "html.parser")

#     # Extract the __VIEWSTATE, __VIEWSTATEGENERATOR, and __EVENTVALIDATION fields
#     viewstate = initial_soup.find("input", {"name": "__VIEWSTATE"})["value"]
#     viewstategenerator = initial_soup.find("input", {"name": "__VIEWSTATEGENERATOR"})["value"]
#     eventvalidation = initial_soup.find("input", {"name": "__EVENTVALIDATION"})["value"]

#     # Prepare the form data
#     form_data = {
#         "__VIEWSTATE": viewstate,
#         "__VIEWSTATEGENERATOR": viewstategenerator,
#         "__EVENTVALIDATION": eventvalidation,
#         "ctl00$MainContent$ddlMonth": f"{month:02d}",
#         "ctl00$MainContent$ddlYear": str(year),
#         "ctl00$MainContent$btnSubmit": "View Interviews",
#     }

#     # Submit the form
#     response = session.post(f"{BASE_URL}default?name={letter}&month={month:02d}&year={year}", data=form_data)
#     soup = BeautifulSoup(response.content, "html.parser")

#     table = soup.find("table", {"id": "MainContent_manyResultsTable"})
#     if not table:
#         return []

#     rows = table.find_all("tr")[1:]  # Skip header row
#     data = []

#     for row in rows:
#         cols = row.find_all("td")
#         if len(cols) >= 8:
#             data.append(
#                 {
#                     "Name": cols[0].text.strip(),
#                     "DIN": cols[1].text.strip(),
#                     "Birth Date": cols[2].text.strip(),
#                     "Race/ethnicity": cols[3].text.strip(),
#                     "Housing or Interview Facility": cols[4].text.strip(),
#                     "Parole Board Interview Date": cols[5].text.strip(),
#                     "Parole Board Interview Type": cols[6].text.strip(),
#                     "Interview Decision": cols[7].text.strip(),
#                 }
#             )

#     return data


# def get_inmate_details(session, din):
#     # Get the inmate details page
#     response = session.get(f"{BASE_URL}default?din={din}")
#     soup = BeautifulSoup(response.content, "html.parser")

#     details = {
#         "DIN": din,
#         "Name": "",
#         "Aggregated Minimum Sentence": "",
#         "Aggregated Maximum Sentence": "",
#         "Release Date": "",
#         "Release Type": "",
#         "Housing/Release Facility": "",
#         "Parole Eligibility Date": "",
#         "Conditional Release Date": "",
#         "Maximum Expiration Date": "",
#         "Parole ME Date": "",
#         "Post Release Supervision ME Date": "",
#         "Parole Board Discharge Date": "",
#         "Crimes": [],
#     }

#     # Parse the parolee information table
#     parolee_table = soup.find("table", {"id": "MainContent_paroleeInformation"})
#     if parolee_table:
#         for row in parolee_table.find_all("tr"):
#             cols = row.find_all("td")
#             if len(cols) == 2:
#                 key = cols[0].text.strip().replace(":", "")
#                 value = cols[1].text.strip()
#                 if key in details:
#                     details[key] = value

#     # Parse the offense information table
#     offense_table = soup.find("table", {"id": "MainContent_offenseInformationTable"})
#     if offense_table:
#         for row in offense_table.find_all("tr")[1:]:  # Skip header row
#             cols = row.find_all("td")
#             if len(cols) == 3:
#                 crime = cols[0].text.strip()
#                 crime_class = cols[1].text.strip()
#                 county = cols[2].text.strip()
#                 details["Crimes"].append(f"{crime} (Class {crime_class}, {county})")

#     details["Crimes"] = "; ".join(details["Crimes"])
#     return details


# def main():
#     years = range(2022, 2025)
#     months = range(1, 13)

#     filename = f"parole_board_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

#     with open(filename, "w", newline="", encoding="utf-8") as csvfile:
#         fieldnames = [
#             "Name",
#             "DIN",
#             "Birth Date",
#             "Race/ethnicity",
#             "Housing or Interview Facility",
#             "Parole Board Interview Date",
#             "Parole Board Interview Type",
#             "Interview Decision",
#             "Aggregated Minimum Sentence",
#             "Aggregated Maximum Sentence",
#             "Release Date",
#             "Release Type",
#             "Housing/Release Facility",
#             "Parole Eligibility Date",
#             "Conditional Release Date",
#             "Maximum Expiration Date",
#             "Parole ME Date",
#             "Post Release Supervision ME Date",
#             "Parole Board Discharge Date",
#             "Crimes",
#         ]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         session = requests.Session()
#         alphabet = string.ascii_uppercase
#         for year in years:
#             for month in months:
#                 for letter in alphabet:
#                     print(f"Scraping data for {year}-{month:02d}, last names starting with {letter}")
#                     data = get_interview_data(session, year, month, letter)
#                     for row in data:
#                         inmate_details = get_inmate_details(session, row["DIN"])
#                         # Combine the data from both sources
#                         combined_data = {**row, **inmate_details}
#                         writer.writerow(combined_data)
#                         time.sleep(1)  # Be polite to the server

#     print(f"Data has been saved to {filename}")


# if __name__ == "__main__":
#     main()


import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import time
import string
import concurrent.futures
import queue
import threading

BASE_URL = "https://publicapps.doccs.ny.gov/ParoleBoardCalendar/"
MAX_WORKERS = 10  # Adjust this based on your system capabilities and ethical considerations
RATE_LIMIT = 0.1  # Minimum time between requests, adjust as needed


class RateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.last_request = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request
            if time_since_last_request < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last_request)
            self.last_request = time.time()


rate_limiter = RateLimiter(RATE_LIMIT)


def get_interview_data(session, year, month, letter):
    rate_limiter.wait()
    try:
        initial_response = session.get(f"{BASE_URL}default")
        initial_soup = BeautifulSoup(initial_response.content, "html.parser")

        viewstate = initial_soup.find("input", {"name": "__VIEWSTATE"})["value"]
        viewstategenerator = initial_soup.find("input", {"name": "__VIEWSTATEGENERATOR"})["value"]
        eventvalidation = initial_soup.find("input", {"name": "__EVENTVALIDATION"})["value"]

        form_data = {
            "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": viewstategenerator,
            "__EVENTVALIDATION": eventvalidation,
            "ctl00$MainContent$ddlMonth": f"{month:02d}",
            "ctl00$MainContent$ddlYear": str(year),
            "ctl00$MainContent$btnSubmit": "View Interviews",
        }

        response = session.post(f"{BASE_URL}default?name={letter}&month={month:02d}&year={year}", data=form_data)
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.find("table", {"id": "MainContent_manyResultsTable"})
        if not table:
            return []

        rows = table.find_all("tr")[1:]
        data = []

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 8:
                data.append(
                    {
                        "Name": cols[0].text.strip(),
                        "DIN": cols[1].text.strip(),
                        "Birth Date": cols[2].text.strip(),
                        "Race/ethnicity": cols[3].text.strip(),
                        "Housing or Interview Facility": cols[4].text.strip(),
                        "Parole Board Interview Date": cols[5].text.strip(),
                        "Parole Board Interview Type": cols[6].text.strip(),
                        "Interview Decision": cols[7].text.strip(),
                    }
                )

        return data
    except Exception as e:
        print(f"Error in get_interview_data: {e}")
        return []


def get_inmate_details(session, din):
    rate_limiter.wait()
    try:
        response = session.get(f"{BASE_URL}default?din={din}")
        soup = BeautifulSoup(response.content, "html.parser")

        details = {
            "DIN": din,
            "Aggregated Minimum Sentence": "",
            "Aggregated Maximum Sentence": "",
            "Release Date": "",
            "Release Type": "",
            "Housing/Release Facility": "",
            "Parole Eligibility Date": "",
            "Conditional Release Date": "",
            "Maximum Expiration Date": "",
            "Parole ME Date": "",
            "Post Release Supervision ME Date": "",
            "Parole Board Discharge Date": "",
            "Crimes": [],
        }

        parolee_table = soup.find("table", {"id": "MainContent_paroleeInformation"})
        if parolee_table:
            for row in parolee_table.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) == 2:
                    key = cols[0].text.strip().replace(":", "")
                    value = cols[1].text.strip()
                    if key in details:
                        details[key] = value

        offense_table = soup.find("table", {"id": "MainContent_offenseInformationTable"})
        if offense_table:
            for row in offense_table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) == 3:
                    crime = cols[0].text.strip()
                    crime_class = cols[1].text.strip()
                    county = cols[2].text.strip()
                    details["Crimes"].append(f"{crime} (Class {crime_class}, {county})")

        details["Crimes"] = "; ".join(details["Crimes"])
        return details
    except Exception as e:
        print(f"Error in get_inmate_details for DIN {din}: {e}")
        return {}


def worker(task_queue, result_queue, session):
    while True:
        task = task_queue.get()
        if task is None:
            break
        year, month, letter = task
        data = get_interview_data(session, year, month, letter)
        for row in data:
            inmate_details = get_inmate_details(session, row["DIN"])
            combined_data = {**row, **inmate_details}
            result_queue.put(combined_data)
        task_queue.task_done()


def main():
    years = range(2023, 2025)
    months = range(1, 13)
    alphabet = string.ascii_uppercase

    filename = f"parole_board_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    fieldnames = [
        "Name",
        "DIN",
        "Birth Date",
        "Race/ethnicity",
        "Housing or Interview Facility",
        "Parole Board Interview Date",
        "Parole Board Interview Type",
        "Interview Decision",
        "Aggregated Minimum Sentence",
        "Aggregated Maximum Sentence",
        "Release Date",
        "Release Type",
        "Housing/Release Facility",
        "Parole Eligibility Date",
        "Conditional Release Date",
        "Maximum Expiration Date",
        "Parole ME Date",
        "Post Release Supervision ME Date",
        "Parole Board Discharge Date",
        "Crimes",
    ]

    task_queue = queue.Queue()
    result_queue = queue.Queue()

    for year in years:
        for month in months:
            for letter in alphabet:
                task_queue.put((year, month, letter))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        sessions = [requests.Session() for _ in range(MAX_WORKERS)]
        for _ in range(MAX_WORKERS):
            executor.submit(worker, task_queue, result_queue, sessions[_])

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                try:
                    result = result_queue.get(timeout=1)
                    writer.writerow(result)
                    result_queue.task_done()
                except queue.Empty:
                    if task_queue.empty() and all(
                        future.done() for future in concurrent.futures.as_completed(executor._threads)
                    ):
                        break

    print(f"Data has been saved to {filename}")


if __name__ == "__main__":
    main()
