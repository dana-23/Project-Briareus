"""
Project Briareus — Settings
Environment config, model parameters, and orchestrator constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
SUPERVISOR_MODEL = "gemini-2.5-pro"
AGENT_MODEL = "gemini-2.0-flash"

SUPERVISOR_TEMPERATURE = 0.0    # greedy supervisor, check facts only
AGENT_TEMPERATURE = 0.4         # slightly higher for creative agent work

# ──────────────────────────────────────────────
# Orchestrator Limits
# ──────────────────────────────────────────────
MAX_ITERATIONS = 10         # total agent calls
MAX_RETRIES_PER_STEP = 2    # retries for a single step
MAX_PLAN_STEPS = 6          # cap on plan complexity

# ──────────────────────────────────────────────
# Tool Configuration
# ──────────────────────────────────────────────
TAVILY_MAX_RESULTS = 10