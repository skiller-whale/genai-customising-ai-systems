
def get_attendance_id(path='/app/sync/attendance_id'):
    """
    Get the attendance ID from a file.
    Args:
        path (str): The path to the file containing the attendance ID.
    Returns:
        str: The attendance ID."""
    with open(path, 'r') as f:
        return f.read().strip()