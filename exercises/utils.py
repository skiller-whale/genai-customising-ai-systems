"""
Utility functions for the GenAI exercises.
"""

import os


def get_attendance_id(env_var='SW_ATTENDANCE_ID'):
    """
    Get the attendance ID from an environment variable.

    Args:
        env_var (str): The environment variable name.
    Returns:
        str: The attendance ID."""

    attendance_id = os.getenv(env_var)

    if not attendance_id:
        raise ValueError(f"Attendance ID not found in environment variable '{env_var}'.")

    return attendance_id
