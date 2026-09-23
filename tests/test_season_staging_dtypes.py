import unittest

import pandas as pd

from nhl_scraper.scraper import NHLScraper


def _sql_kind(series: pd.Series) -> str:
    """Approximate the Postgres type pandas to_sql would infer."""
    if pd.api.types.is_float_dtype(series.dtype):
        return "double precision"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "bigint"
    return "text"


class TestSeasonStagingDtypes(unittest.TestCase):
    def test_sparse_callup_goalie_frame_coerces_hash_args(self):
        # Call-up-only landing scrapes omit many stats. gamesPlayed arrives as
        # int (BIGINT via to_sql); padded nulls become object/TEXT.
        sparse = pd.DataFrame(
            {
                "playerId": ["8484442"],
                "gameTypeId": [2],
                "gamesPlayed": [2],
                "goalsAgainst": [5.0],
                "goalsAgainstAvg": [3.12],
                "leagueAbbrev": ["NHL"],
                "losses": [1],
                "season": [20252026],
                "sequence": [1],
                "shutouts": [0],
                "wins": [1],
                "savePctg": [0.891],
                "shotsAgainst": [46],
                "otLosses": [0],
                "timeOnIce": ["96:14"],
                "teamName.default": ["Toronto Maple Leafs"],
            }
        )

        prepared = NHLScraper._ensure_dataframe_columns(
            sparse,
            NHLScraper.SEASON_GOALIE_STAGING_COLUMNS,
            numeric_columns=NHLScraper.SEASON_GOALIE_NUMERIC_COLUMNS,
            text_columns=NHLScraper.SEASON_GOALIE_TEXT_COLUMNS,
        )

        for column in NHLScraper.SEASON_GOALIE_NUMERIC_COLUMNS:
            self.assertTrue(
                pd.api.types.is_float_dtype(prepared[column]),
                f"{column} should be float, got {prepared[column].dtype}",
            )
            self.assertEqual(_sql_kind(prepared[column]), "double precision")

        # All-null padded columns must still be float, not object/text.
        self.assertTrue(prepared["ties"].isna().all())
        self.assertTrue(prepared["assists"].isna().all())
        self.assertTrue(prepared["gamesStarted"].isna().all())
        self.assertTrue(prepared["goals"].isna().all())
        self.assertTrue(prepared["pim"].isna().all())

        self.assertEqual(prepared["gamesPlayed"].iloc[0], 2.0)
        self.assertEqual(prepared["timeOnIce"].iloc[0], "96:14")
        self.assertEqual(_sql_kind(prepared["timeOnIce"]), "text")
        self.assertEqual(_sql_kind(prepared["leagueAbbrev"]), "text")
        self.assertFalse(pd.api.types.is_numeric_dtype(prepared["timeOnIce"]))

        # These are the args passed to generate_season_goalie_data_hash.
        hash_arg_kinds = [
            _sql_kind(prepared[column])
            for column in (
                "gamesPlayed",
                "goalsAgainst",
                "goalsAgainstAvg",
                "losses",
                "shutouts",
                "ties",
                "wins",
                "assists",
                "gamesStarted",
                "goals",
                "pim",
                "savePctg",
                "shotsAgainst",
                "otLosses",
                "timeOnIce",
            )
        ]
        self.assertEqual(
            hash_arg_kinds,
            ["double precision"] * 14 + ["text"],
            "hash function expects 14 doubles + timeOnIce as text",
        )

    def test_sparse_callup_skater_frame_coerces_numeric_columns(self):
        sparse = pd.DataFrame(
            {
                "playerId": ["8483512"],
                "assists": [1],
                "gameTypeId": [2],
                "gamesPlayed": [3],
                "goals": [0],
                "leagueAbbrev": ["NHL"],
                "season": [20252026],
                "sequence": [1],
                "avgToi": ["12:41"],
                "teamName.default": ["Toronto Maple Leafs"],
            }
        )

        prepared = NHLScraper._ensure_dataframe_columns(
            sparse,
            NHLScraper.SEASON_SKATER_STAGING_COLUMNS,
            numeric_columns=NHLScraper.SEASON_SKATER_NUMERIC_COLUMNS,
            text_columns=NHLScraper.SEASON_SKATER_TEXT_COLUMNS,
        )

        for column in NHLScraper.SEASON_SKATER_NUMERIC_COLUMNS:
            self.assertTrue(
                pd.api.types.is_float_dtype(prepared[column]),
                f"{column} should be float, got {prepared[column].dtype}",
            )

        self.assertTrue(prepared["plusMinus"].isna().all())
        self.assertTrue(prepared["faceoffWinningPctg"].isna().all())
        self.assertTrue(pd.api.types.is_float_dtype(prepared["plusMinus"]))
        self.assertEqual(prepared["avgToi"].iloc[0], "12:41")
        self.assertFalse(pd.api.types.is_numeric_dtype(prepared["avgToi"]))
        self.assertEqual(prepared["gamesPlayed"].iloc[0], 3.0)


if __name__ == "__main__":
    unittest.main()
