import os


def get_metadata(filepath):
    meta = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                content = line.replace('#', '').strip()
                if ':' in content:
                    key, val = content.split(':', 1)
                    meta[key.strip()] = val.strip()
            else:
                break
    return meta


def time_axis_scale(max_time):
    if max_time >= 1e9:
        return 1e9, "10$^9$"
    elif max_time >= 1e8:
        return 1e8, "10$^8$"
    elif max_time >= 1e7:
        return 1e7, "10$^7$"
    elif max_time >= 1e6:
        return 1e6, "10$^6$"
    elif max_time >= 1e5:
        return 1e5, "10$^5$"
    elif max_time >= 1e3:
        return 1e3, "10$^3$"
    else:
        return 1, "1"
