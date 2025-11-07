import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes solo de lectura para Classroom; ajustables si luego lees entregas
SCOPES = ["https://www.googleapis.com/auth/classroom.rosters.readonly"]

def main():
    flow = InstalledAppFlow.from_client_secrets_file(
        "client_secret.json", SCOPES
    )
    creds = flow.run_local_server(port=0)
    # Guarda token para futuras ejecuciones
    with open("token.json", "w") as f:
        f.write(creds.to_json())
    print("✅ OAuth listo. Se guardó token.json")

if __name__ == "__main__":
    main()
