from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import requests

def parse_tasks() -> list[dict[str, str]]:
    load_dotenv()
    lk_cookie = os.getenv("LK_SESSION_COOKIE")

    url = "https://lk.dataschool.yandex.ru/learning/assignments/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; parser/1.0)"
    }

    session = requests.Session()
    if lk_cookie:
        session.cookies.set("sessionid", lk_cookie, domain="lk.dataschool.yandex.ru")

    resp = session.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    try:
        open_tasks_table = soup.find("h3", string="Открытые задания").find_next("table")
    except AttributeError:
        raise RuntimeError("Could not find the open tasks table on the page. Check your cookie")

    tasks = []

    for row in open_tasks_table.find_all("tr", class_="noop"):
        date_block = row.find("div", class_="assignment-date")
        if date_block:
            date = date_block.find("span", class_="nowrap").get_text(strip=True)
            time = date_block.get_text(strip=True).replace(date, "")
            deadline = f"{date} {time.strip()}"
        else:
            deadline = None

        assignment_link = row.find_all("a")[0]
        course_link = row.find_all("a")[1]

        assignment_name = assignment_link.get_text(strip=True)
        course_name = course_link.get_text(strip=True)

        tasks.append({
            "course": course_name,
            "assignment": assignment_name,
            "deadline": deadline
        })

    for t in tasks:
        print(f"{t['course']}: {t['assignment']} — {t['deadline']}")

    return tasks


# TODO: Implement function to get a list of upcoming lectures from learning/timetable/
def parse_lectures() -> list[dict[str, str]]:
    raise NotImplementedError("Function parse_lectures is not yet implemented")
