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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "gemini-2.5-pro")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash")

SUPERVISOR_TEMPERATURE = 0.2    # low temp for reliable routing decisions
AGENT_TEMPERATURE = 0.4         # slightly higher for creative agent work

# ──────────────────────────────────────────────
# Orchestrator Limits
# ──────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))     # total agent calls
MAX_RETRIES_PER_STEP = int(os.getenv("MAX_RETRIES", "2"))   # retries for a single step
MAX_PLAN_STEPS = int(os.getenv("MAX_PLAN_STEPS", "6"))      # cap on plan complexity

# ──────────────────────────────────────────────
# Tool Configuration
# ──────────────────────────────────────────────
TAVILY_MAX_RESULTS = 5