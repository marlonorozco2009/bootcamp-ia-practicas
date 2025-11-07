import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/classroom.rosters.readonly"]
COURSE_ID = "NzgxMTI2MjY3NDgx"

def get_creds_from_token():
    with open("token.json", "r") as f:
        data = json.load(f)
    return Credentials.from_authorized_user_info(data, SCOPES)

def main():
    creds = get_creds_from_token()
    classroom = build("classroom", "v1", credentials=creds)
    resp = classroom.courses().students().list(courseId=COURSE_ID).execute()
    students = resp.get("students", [])
    print(f"👥 Estudiantes en el curso {COURSE_ID}: {len(students)}")
    for s in students:
        nombre = s["profile"]["name"].get("fullName", "")
        email  = s["profile"].get("emailAddress", "")
        print(f"- {nombre} <{email}>")

if __name__ == "__main__":
    main()
