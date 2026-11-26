"""按联赛、比赛、球队和已确认号码查询球员姓名。"""

from __future__ import annotations

import json
from pathlib import Path


class PlayerRoster:
    def __init__(self, records: list[dict] | None = None) -> None:
        self._names: dict[tuple[str, str, str, int], str] = {}
        for record in records or []:
            self.add(record["league"], record["game_id"], record["team"],
                     int(record["number"]), record["name"])

    @classmethod
    def load(cls, path: str | Path) -> "PlayerRoster":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("players", []) if isinstance(data, dict) else data)

    def add(self, league: str, game_id: str, team: str, number: int, name: str) -> None:
        league, game_id, team, name = (str(v).strip() for v in (league, game_id, team, name))
        if not league or not game_id or not team or not name:
            raise ValueError("league/game_id/team/name 不能为空")
        key = (league, game_id, team.casefold(), int(number))
        previous = self._names.get(key)
        if previous is not None and previous != name:
            raise ValueError(f"名单冲突: {key}: {previous} / {name}")
        self._names[key] = name

    def resolve_name(self, league: str, game_id: str, team: str | None,
                     number: int | None) -> str | None:
        if not game_id or not team or number is None:
            return None
        return self._names.get((str(league), str(game_id), str(team).casefold(), int(number)))

    @staticmethod
    def label(team: str | None, number: int | None, name: str | None) -> str:
        if name:
            return name
        prefix = f"{team} " if team else ""
        return f"{prefix}#{number}" if number is not None else prefix.rstrip() or "unknown"

    def __len__(self) -> int:
        return len(self._names)