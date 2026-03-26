import logging
import os
import time

import requests
import tqdm
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("ace.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ace_lib")

brain_api_url = "https://api.worldquantbrain.com"


class SingleSession(requests.Session):
    """A wrapper for requests.Session as expected by ace_extensions."""
    pass


def _request_with_retry(s, method, url, **kwargs):
    """
    Makes a request and automatically retries on 429 rate limit.
    Waits for Retry-After seconds before retrying.
    """
    while True:
        response = getattr(s, method)(url, **kwargs)
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 10))
            logger.warning(f"Rate limited (429). Waiting {retry_after}s before retrying...")
            time.sleep(retry_after)
            continue
        return response


def start_session():
    """Logs into the BRAIN platform using credentials from .env"""
    s = SingleSession()
    email = os.getenv("BRAIN_CREDENTIAL_EMAIL")
    password = os.getenv("BRAIN_CREDENTIAL_PASSWORD")

    if not email or not password:
        logger.error("Credentials missing in .env file.")
        return s

    # BRAIN uses HTTP Basic Auth for authentication
    response = _request_with_retry(
        s, "post",
        f"{brain_api_url}/authentication",
        auth=HTTPBasicAuth(email.strip(), password.strip()),
    )

    if response.status_code == 201:
        logger.info("Successfully authenticated with WorldQuant BRAIN.")
    elif response.status_code == 400:
        logger.error(f"Bad request during authentication: {response.text}")
    elif response.status_code == 401:
        logger.error(f"Authentication failed. Wrong email or password. Details: {response.text}")
    else:
        logger.error(f"Unexpected authentication response: {response.status_code} - {response.text}")

    return s


def check_session_timeout(s):
    """
    Checks if the current session is still valid.
    Returns seconds remaining until expiry (0 if expired or error).
    """
    try:
        response = _request_with_retry(s, "get", f"{brain_api_url}/authentication")
        if response.status_code == 200:
            data = response.json()
            expiry = (
                data.get("expiresAt")
                or data.get("expires")
                or data.get("expiry")
                or 0
            )
            if expiry:
                remaining = float(expiry) - time.time()
                return max(remaining, 0)
        return 0
    except Exception as e:
        logger.warning(f"check_session_timeout error: {e}")
        return 0


def check_session_and_relogin(s):
    """
    Ensures the session is active before making requests.
    Re-authenticates if session is expiring within 60 seconds.
    """
    ttl = check_session_timeout(s)
    if ttl < 60:
        logger.info("Session expired or expiring soon. Re-authenticating...")
        return start_session()
    return s


def start_simulation(s, alpha_data):
    """
    Sends an alpha expression to the /simulations endpoint.
    Returns the response object (contains Location header for polling).
    """
    response = _request_with_retry(
        s, "post",
        f"{brain_api_url}/simulations",
        json=alpha_data,
    )

    if response.status_code not in (200, 201):
        logger.error(f"Failed to start simulation. Status: {response.status_code}. Details: {response.text}")

    return response


def simulation_progress(s, simulation_response):
    """
    Polls the simulation progress URL until complete.
    Returns dict: {"completed": bool, "result": alpha_json or None}
    """
    if "Location" not in simulation_response.headers:
        logger.error("No Location header found. Simulation likely failed to start.")
        return {"completed": False, "result": None}

    location = simulation_response.headers["Location"]
    progress_url = location if location.startswith("http") else brain_api_url + location

    # Poll until Retry-After header disappears (simulation complete)
    while True:
        progress = _request_with_retry(s, "get", progress_url)

        if "Retry-After" not in progress.headers:
            break

        time.sleep(float(progress.headers["Retry-After"]))

    data = progress.json()

    # Check for failure statuses
    status = data.get("status", "")
    if status in ["ERROR", "FAIL", "TIMEOUT", "CANCELLED"]:
        logger.warning(
            f"Simulation ended with status: {status}. "
            f"Message: {data.get('message', '')}"
        )
        return {"completed": False, "result": None}

    # Extract alpha ID from completed simulation
    alpha_id = data.get("alpha")
    if not alpha_id:
        logger.error("Simulation completed but no alpha ID returned.")
        return {"completed": False, "result": None}

    # Fetch the full alpha result JSON
    alpha_result = _request_with_retry(s, "get", f"{brain_api_url}/alphas/{alpha_id}")

    if alpha_result.status_code != 200:
        logger.error(f"Failed to fetch alpha {alpha_id}. Details: {alpha_result.text}")
        return {"completed": False, "result": None}

    return {
        "completed": True,
        "result": alpha_result.json(),
    }


def get_simulation_result_json(s, alpha_id):
    """
    Retrieves the full alpha JSON object by alpha ID.
    Contains code, settings, and in-sample stats.
    """
    response = _request_with_retry(s, "get", f"{brain_api_url}/alphas/{alpha_id}")

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to fetch alpha {alpha_id}. Details: {response.text}")
        return None