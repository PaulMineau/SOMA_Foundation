"""ResMed myAir API client — fetches daily CPAP summary data.

Uses Okta OAuth2 with PKCE. Reads credentials from ~/.resmed_config.json.
Tokens cached in ~/.resmed_tokens.json.

Setup:
  1. Create ~/.resmed_config.json:
     {"email": "you@example.com", "password": "your_myair_password", "country": "US"}
  2. Run: python -m soma.proto_self.cpap.myair_client
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urldefrag

import httpx
import jwt

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.expanduser("~/.resmed_config.json")
TOKEN_FILE = os.path.expanduser("~/.resmed_tokens.json")

# Okta endpoints (NA region)
OKTA_HOST = "resmed-ext-1.okta.com"
AUTH_SERVER = "aus4ccsxvnidQgLmA297"
CLIENT_ID = "0oa4ccq1v413ypROi297"
REDIRECT_URI = "https://myair.resmed.com"

# GraphQL endpoint
GRAPHQL_URL = "https://graphql.myair-prd.dht.live/graphql"
API_KEY = "da2-cenztfjrezhwphdqtwtbpqvzui"

# Static headers
HANDSET_HEADERS = {
    "rmdhandsetid": "02c1c662-c289-41fd-a9ae-196ff15b5166",
    "rmdlanguage": "en",
    "rmdhandsetmodel": "Chrome",
    "rmdhandsetosversion": "127.0.6533.119",
    "rmdproduct": "myAir",
    "rmdappversion": "1.0.0",
    "rmdhandsetplatform": "Web",
}


def _load_config() -> dict[str, str]:
    """Load myAir credentials from ~/.resmed_config.json."""
    if not os.path.exists(CONFIG_FILE):
        template = {
            "email": "YOUR_MYAIR_EMAIL",
            "password": "YOUR_MYAIR_PASSWORD",
            "country": "US",
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(template, f, indent=2)
        raise RuntimeError(
            f"Created config template at {CONFIG_FILE}. "
            "Fill in your myAir email/password and re-run."
        )

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    if "YOUR_MYAIR" in config.get("email", ""):
        raise RuntimeError(f"Edit {CONFIG_FILE} with real credentials first.")

    return config


def _save_tokens(tokens: dict) -> None:
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def _load_tokens() -> dict | None:
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE) as f:
        return json.load(f)


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier + code_challenge."""
    raw = secrets.token_bytes(40)
    verifier = base64.urlsafe_b64encode(raw).decode().rstrip("=").replace("_", "").replace("-", "")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    return verifier, challenge


class MyAirClient:
    """ResMed myAir API client."""

    def __init__(self) -> None:
        self.config = _load_config()
        self.tokens: dict | None = _load_tokens()
        self._country_id: str | None = None

    async def authenticate(self) -> dict:
        """Run the full Okta OAuth2 + PKCE authentication flow."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            # Step 1: Grab DT cookie (expect 400)
            try:
                await client.get(f"https://{OKTA_HOST}/oauth2/{AUTH_SERVER}/v1/authorize")
            except httpx.HTTPError:
                pass

            # Step 2: Login with username/password
            resp = await client.post(
                f"https://{OKTA_HOST}/api/v1/authn",
                json={
                    "username": self.config["email"],
                    "password": self.config["password"],
                },
            )
            resp.raise_for_status()
            authn = resp.json()

            if authn.get("status") == "MFA_REQUIRED":
                raise RuntimeError(
                    "MFA required on your myAir account. Standalone MFA support "
                    "not yet implemented — disable MFA in account settings, or request support."
                )

            session_token = authn.get("sessionToken")
            if not session_token:
                raise RuntimeError(f"Unexpected authn response: {authn}")

            # Step 3: PKCE
            verifier, challenge = _generate_pkce()

            # Step 4: Authorization code via authorize endpoint
            resp = await client.get(
                f"https://{OKTA_HOST}/oauth2/{AUTH_SERVER}/v1/authorize",
                params={
                    "client_id": CLIENT_ID,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "prompt": "none",
                    "redirect_uri": REDIRECT_URI,
                    "response_mode": "fragment",
                    "response_type": "code",
                    "sessionToken": session_token,
                    "scope": "openid profile email",
                    "state": secrets.token_hex(16),
                },
            )

            if resp.status_code not in (302, 303):
                raise RuntimeError(f"Expected redirect from authorize, got {resp.status_code}")

            location = resp.headers.get("location", "")
            _, fragment = urldefrag(location)
            params = parse_qs(fragment)
            code = params.get("code", [None])[0]
            if not code:
                raise RuntimeError(f"No code in redirect: {location}")

            # Step 5: Exchange code for tokens
            resp = await client.post(
                f"https://{OKTA_HOST}/oauth2/{AUTH_SERVER}/v1/token",
                data={
                    "client_id": CLIENT_ID,
                    "redirect_uri": REDIRECT_URI,
                    "grant_type": "authorization_code",
                    "code_verifier": verifier,
                    "code": code,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            tokens = resp.json()

        # Decode id_token to extract myAirCountryId
        id_token = tokens.get("id_token", "")
        decoded = jwt.decode(id_token, options={"verify_signature": False})
        self._country_id = decoded.get("myAirCountryId", "US")
        tokens["country_id"] = self._country_id
        tokens["cached_at"] = datetime.now().isoformat()

        self.tokens = tokens
        _save_tokens(tokens)
        logger.info("myAir authentication successful, country=%s", self._country_id)
        return tokens

    async def _ensure_auth(self) -> None:
        """Ensure we have a valid access token."""
        if self.tokens is None:
            await self.authenticate()
            return

        cached = self.tokens.get("cached_at")
        if cached:
            try:
                cached_dt = datetime.fromisoformat(cached)
                expires_in = self.tokens.get("expires_in", 3600)
                if datetime.now() - cached_dt < timedelta(seconds=expires_in - 300):
                    self._country_id = self.tokens.get("country_id", "US")
                    return
            except ValueError:
                pass

        await self.authenticate()

    async def get_sleep_records(
        self,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> list[dict]:
        """Fetch sleep records (daily summaries) for a date range.

        Dates are YYYY-MM-DD strings. If not provided, fetches last 30 days.
        Returns a list of daily records with fields:
          startDate, totalUsage (min), sleepScore, ahi, leakPercentile, maskPairCount, etc.
        """
        await self._ensure_auth()

        if start_month is None:
            start_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if end_month is None:
            end_month = datetime.now().strftime("%Y-%m-%d")

        country = self._country_id or self.config.get("country", "US")

        query = f"""
        query GetPatientSleepRecords {{
          getPatientWrapper {{
            patient {{ firstName }}
            sleepRecords(startMonth: "{start_month}", endMonth: "{end_month}") {{
              items {{
                startDate totalUsage sleepScore usageScore ahiScore
                maskScore leakScore ahi maskPairCount leakPercentile
                sleepRecordPatientId __typename
              }}
            }}
          }}
        }}
        """

        headers = {
            "x-api-key": API_KEY,
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "rmdcountry": country,
            "Content-Type": "application/json",
            **HANDSET_HEADERS,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                GRAPHQL_URL,
                headers=headers,
                json={"query": query, "operationName": "GetPatientSleepRecords"},
            )
            resp.raise_for_status()
            data = resp.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        records = (
            data.get("data", {})
            .get("getPatientWrapper", {})
            .get("sleepRecords", {})
            .get("items", [])
        )
        return records


async def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    client = MyAirClient()
    records = await client.get_sleep_records()

    print(f"\nFetched {len(records)} sleep records from myAir\n")
    for r in records[-7:]:  # Last 7 days
        usage_hrs = r.get("totalUsage", 0) / 60
        print(
            f"  {r['startDate']}  "
            f"AHI: {r.get('ahi', 0):.1f}  "
            f"Usage: {usage_hrs:.1f}h  "
            f"Score: {r.get('sleepScore', 0)}  "
            f"Leak %ile: {r.get('leakPercentile', 0)}"
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
