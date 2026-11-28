"""NBA / FIBA 球队常见主场配色，用于合成球场漆区和替补席。

颜色取各队广泛使用的品牌主色/辅色的近似值；不包含队徽，避免引入商标贴图。
"""

TEAM_PALETTES = [
    # NBA
    {"name": "Lakers", "model": "NBA", "primary": "#552583", "secondary": "#FDB927"},
    {"name": "Celtics", "model": "NBA", "primary": "#007A33", "secondary": "#FFFFFF"},
    {"name": "Bulls", "model": "NBA", "primary": "#CE1141", "secondary": "#111111"},
    {"name": "Warriors", "model": "NBA", "primary": "#1D428A", "secondary": "#FFC72C"},
    {"name": "Knicks", "model": "NBA", "primary": "#006BB6", "secondary": "#F58426"},
    {"name": "Heat", "model": "NBA", "primary": "#98002E", "secondary": "#F9A01B"},
    {"name": "Spurs", "model": "NBA", "primary": "#111111", "secondary": "#C4CED4"},
    {"name": "Mavericks", "model": "NBA", "primary": "#00538C", "secondary": "#FFFFFF"},
    {"name": "Rockets", "model": "NBA", "primary": "#CE1141", "secondary": "#111111"},
    {"name": "76ers", "model": "NBA", "primary": "#006BB6", "secondary": "#ED174C"},
    {"name": "Bucks", "model": "NBA", "primary": "#00471B", "secondary": "#EEE1C6"},
    {"name": "Suns", "model": "NBA", "primary": "#E56020", "secondary": "#1D1160"},
    {"name": "Nuggets", "model": "NBA", "primary": "#0E2240", "secondary": "#FEC524"},
    {"name": "Clippers", "model": "NBA", "primary": "#C8102E", "secondary": "#1D428A"},
    {"name": "Raptors", "model": "NBA", "primary": "#77233F", "secondary": "#CE1141"},
    {"name": "Thunder", "model": "NBA", "primary": "#007AC1", "secondary": "#EF3B2D"},
    {"name": "Trail Blazers", "model": "NBA", "primary": "#E03A3E", "secondary": "#111111"},
    {"name": "Kings", "model": "NBA", "primary": "#5A2D81", "secondary": "#63727A"},
    {"name": "Grizzlies", "model": "NBA", "primary": "#5D76A9", "secondary": "#12173F"},
    {"name": "Pelicans", "model": "NBA", "primary": "#0C2340", "secondary": "#C8102E"},
    {"name": "Timberwolves", "model": "NBA", "primary": "#0C2340", "secondary": "#236192"},
    {"name": "Hornets", "model": "NBA", "primary": "#1D1160", "secondary": "#00788C"},
    {"name": "Hawks", "model": "NBA", "primary": "#E03A3E", "secondary": "#C1D32F"},
    {"name": "Magic", "model": "NBA", "primary": "#0077C0", "secondary": "#111111"},
    {"name": "Pistons", "model": "NBA", "primary": "#C8102E", "secondary": "#1D42BA"},
    {"name": "Cavaliers", "model": "NBA", "primary": "#860038", "secondary": "#FDBB30"},
    {"name": "Pacers", "model": "NBA", "primary": "#002D62", "secondary": "#FDBB30"},
    {"name": "Wizards", "model": "NBA", "primary": "#002B5C", "secondary": "#E31837"},
    {"name": "Nets", "model": "NBA", "primary": "#111111", "secondary": "#FFFFFF"},
    {"name": "Jazz", "model": "NBA", "primary": "#002B5C", "secondary": "#F9A01B"},
    # FIBA / 国家队常见配色
    {"name": "Spain", "model": "FIBA", "primary": "#C60B1E", "secondary": "#FFC400"},
    {"name": "France", "model": "FIBA", "primary": "#0055A4", "secondary": "#FFFFFF"},
    {"name": "Serbia", "model": "FIBA", "primary": "#C6363C", "secondary": "#0C4076"},
    {"name": "Greece", "model": "FIBA", "primary": "#0D5EAF", "secondary": "#FFFFFF"},
    {"name": "Lithuania", "model": "FIBA", "primary": "#FDB913", "secondary": "#006A44"},
    {"name": "Slovenia", "model": "FIBA", "primary": "#005A9C", "secondary": "#FFFFFF"},
    {"name": "Australia", "model": "FIBA", "primary": "#00843D", "secondary": "#FFCD00"},
    {"name": "Canada", "model": "FIBA", "primary": "#D80621", "secondary": "#FFFFFF"},
    {"name": "Argentina", "model": "FIBA", "primary": "#74ACDF", "secondary": "#FFFFFF"},
    {"name": "Brazil", "model": "FIBA", "primary": "#FFDF00", "secondary": "#009C3B"},
    {"name": "Turkey", "model": "FIBA", "primary": "#E30A17", "secondary": "#FFFFFF"},
    {"name": "Germany", "model": "FIBA", "primary": "#111111", "secondary": "#DD0000"},
    {"name": "Italy", "model": "FIBA", "primary": "#0066B3", "secondary": "#FFFFFF"},
    {"name": "Croatia", "model": "FIBA", "primary": "#E31B23", "secondary": "#FFFFFF"},
]


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


__all__ = ["TEAM_PALETTES", "hex_rgb"]
