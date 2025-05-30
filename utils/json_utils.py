import datetime
import collections.abc

def convert_datetime_to_iso_string(obj):
    """
    Recursively convert datetime objects in a dictionary or list to ISO 8601 strings.
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, collections.abc.Mapping):
        return {k: convert_datetime_to_iso_string(v) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes)):
        return [convert_datetime_to_iso_string(elem) for elem in obj]
    else:
        return obj 