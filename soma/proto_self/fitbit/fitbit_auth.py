"""
fitbit_auth.py — One-time OAuth2 setup for Fitbit Personal app.

Run this ONCE to authenticate and save your tokens to .fitbit_tokens.json.
After that, fitbit_client.py handles automatic token refresh.

Prerequisites:
  1. Register a "Personal" app at https://dev.fitbit.com/apps/new
     - OAuth 2.0 Application Type: Personal
     - Callback URL: http://localhost:8080
  2. Copy your CLIENT_ID and CLIENT_SECRET into .fitbit_config.json (see below)
  3. pip install requests requests-oauthlib

Usage:
  python fitbit_auth.py
"""

import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import requests
from requests_oauthlib import OAuth2Session

CONFIG_FILE = os.path.expanduser("~/.fitbit_config.json")
TOKEN_FILE = os.path.expanduser("~/.fitbit_tokens.json")

AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
REDIRECT_URI = "http://localhost:8080"

SCOPES = [
    "heartrate",
    "sleep",
    "activity",
    "profile",
    "oxygen_saturation",
    "respiratory_rate",
]

# ── Config bootstrap ──────────────────────────────────────────────────────────

def bootstrap_config():
    """Create config file template if it doesn't exist."""
    if not os.path.exists(CONFIG_FILE):
        template = {
            "client_id": "YOUR_CLIENT_ID_HERE",
            "client_secret": "YOUR_CLIENT_SECRET_HERE",
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(template, f, indent=2)
        print(f"\n📁 Created config template at {CONFIG_FILE}")
        print("   → Fill in your CLIENT_ID and CLIENT_SECRET from dev.fitbit.com")
        print("   → Then re-run this script.\n")
        sys.exit(0)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    if "YOUR_CLIENT_ID" in config["client_id"]:
        print(f"\n⚠️  Edit {CONFIG_FILE} and add your real CLIENT_ID / CLIENT_SECRET first.\n")
        sys.exit(1)

    return config


# ── Local callback server ─────────────────────────────────────────────────────

captured_code = {}


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if "code" in params:
            captured_code["code"] = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h2>Auth complete. You can close this tab.</h2>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h2>No code received. Check the URL.</h2>")

    def log_message(self, format, *args):
        pass  # suppress request logging


def run_callback_server():
    server = HTTPServer(("localhost", 8080), CallbackHandler)
    server.handle_request()  # blocks until one request is received


# ── OAuth flow ────────────────────────────────────────────────────────────────

def authenticate(config):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # localhost only

    oauth = OAuth2Session(
        client_id=config["client_id"],
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
    )

    auth_url, state = oauth.authorization_url(AUTHORIZE_URL)

    print("\n🌐 Opening Fitbit authorization page in your browser...")
    print(f"   If it doesn't open, visit:\n   {auth_url}\n")
    webbrowser.open(auth_url)

    print("⏳ Waiting for callback on http://localhost:8080 ...")
    run_callback_server()

    if "code" not in captured_code:
        print("❌ No authorization code received. Exiting.")
        sys.exit(1)

    # Exchange code for tokens using Basic Auth (required by Fitbit)
    from requests.auth import HTTPBasicAuth

    token = oauth.fetch_token(
        TOKEN_URL,
        code=captured_code["code"],
        auth=HTTPBasicAuth(config["client_id"], config["client_secret"]),
    )

    with open(TOKEN_FILE, "w") as f:
        json.dump(token, f, indent=2)

    print(f"\n✅ Tokens saved to {TOKEN_FILE}")
    print("   Run fitbit_client.py or soma_fitbit_ingestor.py next.\n")
    return token


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = bootstrap_config()

    if os.path.exists(TOKEN_FILE):
        print(f"ℹ️  Tokens already exist at {TOKEN_FILE}")
        resp = input("   Re-authenticate? [y/N] ").strip().lower()
        if resp != "y":
            print("   Skipping. Delete the token file to force re-auth.")
            sys.exit(0)

    authenticate(config)
