from supabase.client import create_client, Client
from typing import Any, Optional
import os
from typing import List

import logging
logging.getLogger("httpx").setLevel(logging.WARNING) # Disable Supabase info logs

# Auth

def auth():
    url: Optional[str] = os.environ.get("SUPABASE_PUBLIC_URL")
    key: Optional[str] = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    # Check if either url or key is None and raise an error if so
    if url is None:
        raise ValueError("Environment variable 'SUPABASE_PUBLIC_URL' not found.")
    if key is None:
        raise ValueError("Environment variable 'SUPABASE_SERVICE_ROLE_KEY' not found.")

    # Continue with logic, as url and key are now guaranteed to be non-None
    supabase: Client = create_client(url, key)
    
    return supabase

# Fetch data from each table
def fetchTables(supabase: Client, getAppUsageData: bool = True):
    _acceptance_data = supabase.table("_acceptance").select("*").execute().data
    _actions_data = supabase.table("_actions").select("*").execute().data
    _app_names_data = supabase.table("_app_names").select("*").execute().data
    _location_data = supabase.table("_location").select("*").execute().data
    _sex = supabase.table("_sex").select("*").execute().data
    _weekdays = supabase.table("_weekdays").select("*").execute().data

    user_app_usage_data: List[Any] = []
    
    if (getAppUsageData):
        user_app_usage_data = supabase.table("user_app_usage").select("*").execute().data
    
    users_data = supabase.table("users").select("*").execute().data
        
    return _acceptance_data, _actions_data, _app_names_data, _location_data, _sex, _weekdays, user_app_usage_data, users_data

