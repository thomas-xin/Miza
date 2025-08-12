from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Any, Literal

# --- Durations (equivalent to Luxon Durations) ---
LAND_OFFSET = timedelta(minutes=8, seconds=40)
END_OFFSET = timedelta(hours=4)

BLACK_SHARD_INTERVAL = timedelta(hours=8)
RED_SHARD_INTERVAL = timedelta(hours=6)

# Realms (index-based selection)
REALMS = ("prairie", "forest", "valley", "wasteland", "vault")

# Number of map variants (default = 1 if not present)
NUM_MAP_VARIANTS: Dict[str, int] = {
    'prairie.butterfly': 3,
    'prairie.village': 3,
    'prairie.bird': 2,
    'prairie.island': 3,
    'forest.brook': 2,
    'forest.end': 2,
    'valley.rink': 3,
    'valley.dreams': 2,
    'wasteland.temple': 3,
    'wasteland.battlefield': 3,
    'wasteland.graveyard': 2,
    'wasteland.crab': 2,
    'wasteland.ark': 4,
    'vault.starlight': 3,
    'vault.jelly': 2,
}

# Override reward AC (only applies for red shards)
OVERRIDE_REWARD_AC: Dict[str, float] = {
    'forest.end': 2.5,
    'valley.dreams': 2.5,
    'forest.tree': 3.5,
    'vault.jelly': 3.5,
}

@dataclass(frozen=True)
class ShardConfig:
    no_shard_weekdays: List[int]      # ISO weekday numbers with no shard (1=Mon ... 7=Sun)
    interval: timedelta
    offset: timedelta
    maps: tuple[str, str, str, str, str]
    def_reward_ac: Optional[float] = None  # Only relevant for red shards


# Mirrors the TypeScript shardsInfo array (order matters: indexes 0..4)
SHARDS_INFO: List[ShardConfig] = [
    ShardConfig(
        no_shard_weekdays=[6, 7],  # Sat, Sun
        interval=BLACK_SHARD_INTERVAL,
        offset=timedelta(hours=1, minutes=50),
        maps=('prairie.butterfly', 'forest.brook', 'valley.rink', 'wasteland.temple', 'vault.starlight'),
    ),
    ShardConfig(
        no_shard_weekdays=[7, 1],  # Sun, Mon
        interval=BLACK_SHARD_INTERVAL,
        offset=timedelta(hours=2, minutes=10),
        maps=('prairie.village', 'forest.boneyard', 'valley.rink', 'wasteland.battlefield', 'vault.starlight'),
    ),
    ShardConfig(
        no_shard_weekdays=[1, 2],  # Mon, Tue
        interval=RED_SHARD_INTERVAL,
        offset=timedelta(hours=7, minutes=40),
        maps=('prairie.cave', 'forest.end', 'valley.dreams', 'wasteland.graveyard', 'vault.jelly'),
        def_reward_ac=2.0,
    ),
    ShardConfig(
        no_shard_weekdays=[2, 3],  # Tue, Wed
        interval=RED_SHARD_INTERVAL,
        offset=timedelta(hours=2, minutes=20),
        maps=('prairie.bird', 'forest.tree', 'valley.dreams', 'wasteland.crab', 'vault.jelly'),
        def_reward_ac=2.5,
    ),
    ShardConfig(
        no_shard_weekdays=[3, 4],  # Wed, Thu
        interval=RED_SHARD_INTERVAL,
        offset=timedelta(hours=3, minutes=30),
        maps=('prairie.island', 'forest.sunny', 'valley.hermit', 'wasteland.ark', 'vault.jelly'),
        def_reward_ac=3.5,
    ),
]


@dataclass
class Occurrence:
    start: datetime
    land: datetime
    end: datetime


@dataclass
class ShardInfo:
    date: datetime                      # original input date (may be any tz)
    is_red: bool
    has_shard: bool
    offset: timedelta
    interval: timedelta
    last_end: datetime
    realm: str
    map: str
    num_variant: int
    reward_ac: Optional[float]
    occurrences: List[Occurrence]
    was_override: bool


# Type of override parameter
Override = Dict[str, Any]  # keys: isRed, realm, group, hasShard, map


def _is_dst(dt: datetime) -> bool:
    """Return True if the datetime is in daylight saving time."""
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware for DST check.")
    delta = dt.dst()
    return bool(delta and delta != timedelta(0))


def get_shard_info(date: datetime, override: Optional[Override] = None) -> ShardInfo:
    """
    Python translation of the TypeScript getShardInfo.
    date: a datetime (naive = assumed UTC; aware = any timezone)
    override: optional dict with keys:
        isRed: bool
        realm: int (0..4)
        group: int (0..4) index into SHARDS_INFO
        hasShard: bool
        map: str
    """
    if date.tzinfo is None:
        # Assume input naive datetime is UTC, mimic typical behavior
        date = date.replace(tzinfo=ZoneInfo("UTC"))

    # Convert to America/Los_Angeles and truncate to start of local day
    la_tz = ZoneInfo("America/Los_Angeles")
    date_la = date.astimezone(la_tz)
    today = datetime(year=date_la.year, month=date_la.month, day=date_la.day, tzinfo=la_tz)

    day_of_month = today.day
    day_of_week = today.isoweekday()  # 1=Mon ... 7=Sun (matches Luxon)

    is_red = override.get("isRed") if override and "isRed" in override else (day_of_month % 2 == 1)
    realm_idx = override.get("realm") if override and "realm" in override else (day_of_month - 1) % 5
    # group selection logic from TS:
    if override and "group" in override:
        info_index = override["group"]
    else:
        if day_of_month % 2 == 1:
            # odd day
            info_index = int(((day_of_month - 1) / 2) % 3) + 2
        else:
            # even day
            info_index = int((day_of_month / 2) % 2)

    shard_cfg = SHARDS_INFO[info_index]

    has_shard = override.get("hasShard") if override and "hasShard" in override else (day_of_week not in shard_cfg.no_shard_weekdays)
    selected_map = override.get("map") if override and "map" in override else shard_cfg.maps[realm_idx]

    reward_ac: Optional[float] = None
    if is_red:
        reward_ac = OVERRIDE_REWARD_AC.get(selected_map, shard_cfg.def_reward_ac)

    num_variant = NUM_MAP_VARIANTS.get(selected_map, 1)

    # First start = start of day + offset
    first_start = today + shard_cfg.offset

    # DST adjustment logic (same semantics as TS code)
    # Detect timezone change between midnight and first_start (only on Sunday)
    if day_of_week == 7 and _is_dst(today) != _is_dst(first_start):
        if _is_dst(first_start):
            # If first_start is in DST but today isn't, subtract 1 hour
            first_start = first_start - timedelta(hours=1)
        else:
            # If first_start not in DST but today is, add 1 hour
            first_start = first_start + timedelta(hours=1)

    occurrences: List[Occurrence] = []
    for i in range(3):
        start_i = first_start + shard_cfg.interval * i
        occurrences.append(
            Occurrence(
                start=start_i,
                land=start_i + LAND_OFFSET,
                end=start_i + END_OFFSET,
            )
        )

    return ShardInfo(
        date=date,
        is_red=is_red,
        has_shard=has_shard,
        offset=shard_cfg.offset,
        interval=shard_cfg.interval,
        last_end=occurrences[2].end,
        realm=REALMS[realm_idx],
        map=selected_map,
        num_variant=num_variant,
        reward_ac=reward_ac,
        occurrences=occurrences,
        was_override=bool(override),
    )


def find_next_shard(from_dt: datetime, only: Optional[Literal["black", "red"]] = None) -> ShardInfo:
    """
    Recursively (iteratively) find the next shard info whose window still includes 'from_dt'.
    only: filter by color ('black' or 'red') or None for any.
    """
    # We'll avoid deep recursion by looping.
    probe = from_dt
    while True:
        info = get_shard_info(probe)
        color_match = True
        if only == "red":
            color_match = info.is_red
        elif only == "black":
            color_match = not info.is_red

        if info.has_shard and probe < info.last_end and color_match:
            return info
        # Move to next day (keep same timezone if aware)
        probe = (probe + timedelta(days=1)).replace(tzinfo=probe.tzinfo)


# --- Example usage (remove or guard under __main__ as needed) ---
if __name__ == "__main__":
    now = datetime.now(ZoneInfo("UTC"))
    shard = get_shard_info(now)
    print("Today shard info:")
    print(f"  Realm: {shard.realm}")
    print(f"  Map: {shard.map}")
    print(f"  Red?: {shard.is_red}")
    print(f"  Reward AC: {shard.reward_ac}")
    for i, occ in enumerate(shard.occurrences, start=1):
        print(f"  Occurrence {i}: start={occ.start}, land={occ.land}, end={occ.end}")

    next_red = find_next_shard(now, only="red")
    print("\nNext red shard date (local LA day):", next_red.occurrences[0].start.astimezone(ZoneInfo("America/Los_Angeles")).date())