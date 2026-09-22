import requests
import pandas as pd
from time import sleep
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text

NHL_SCHEDULE_TZ = ZoneInfo("America/New_York")

class NHLScraper:
    def __init__(self):
        self.base_url = "https://api.nhle.com/stats/rest/en"
        self.web_api_url = "https://api-web.nhle.com/v1"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.active_team_codes = [
            'ANA', 'UTA', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 
            'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 
            'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 
            'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG'
        ]
        
        self.game_types = [2, 3]  # 2 for regular season, 3 for playoffs

    def get_current_season(self) -> str:
        """Calculate the current NHL season string (September start date)."""
        current_date = datetime.now()
        current_year = current_date.year
        if current_date.month >= 9:
            season = f"{current_year}{current_year + 1}"
        else:
            season = f"{current_year - 1}{current_year}"
        return season

    def get_all_teams(self) -> pd.DataFrame:
        """
        Fetch all teams and their details using the /team endpoint.
        Returns a list of team dictionaries.
        """
        url = f"{self.base_url}/team"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("data", []))

    def get_team_summary(self, current_season_only: bool = True, game_type: int = 2) -> pd.DataFrame:
        """
        Fetch team summary statistics (wins, losses, goals for/against, etc) 
        for all teams and seasons.
        
        Args:
            current_season_only: If True, only fetch current season data. 
                                If False, fetch all seasons. Defaults to True.
            game_type: Game type to filter by. 2 for regular season, 3 for playoffs. Defaults to 2.
        """
        url = f"{self.base_url}/team/summary"
        season_id = self.get_current_season()  # returns str like "20232024"
        limit = 50
        all_data = []
        offset = 0

        while True:
            params = {
                "limit": limit,
                "start": offset,
                "sort": "seasonId",
            }
            
            # Build cayenneExp filter
            if current_season_only:
                params["cayenneExp"] = f"seasonId={season_id} and gameTypeId={game_type}"
            else:
                params["cayenneExp"] = f"gameTypeId={game_type}"
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('data'):
                    break
                    
                all_data.extend(data['data'])
                
                if len(data['data']) < limit:
                    break
                    
                offset += limit
                
            except requests.RequestException as e:
                self.logger.error(f"Error fetching team summary: {e}")
                break

        df = pd.DataFrame(all_data)
        # Add gameType column if not already present
        if not df.empty and 'gameTypeId' not in df.columns:
            df['gameTypeId'] = game_type
        
        return df
    
    def get_all_drafts(self) -> pd.DataFrame:
        """"""
        url = f"{self.base_url}/draft"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("data", []))

    def get_all_franchises(self) -> pd.DataFrame:
        """
        Fetch all franchises and their details using the /franchise endpoint.
        Returns a DataFrame of franchise data.
        """
        url = f"{self.base_url}/franchise"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("data", []))
    
    async def scrape_all_drafts_async(self) -> pd.DataFrame:
        """Scrape draft data for all available years and return as DataFrame."""
        try:
            draft_df = self.get_all_drafts()
            urls = [
                f"https://api-web.nhle.com/v1/draft/picks/{draft}/all" 
                for draft in draft_df['draftYear'].to_list()
            ]
            
            responses = await self._fetch_all_data(urls)
            
            # Process responses into a unified DataFrame
            all_picks = []
            for response, draft_year in zip(responses, draft_df['draftYear']):
                if response and 'picks' in response:
                    picks = response['picks']
                    # Add draft year to each pick
                    for pick in picks:
                        pick['draftYear'] = draft_year
                        for key, value in pick.items():
                            if isinstance(value, dict) and 'default' in value:
                                pick[key] = value['default']

                    all_picks.extend(picks)
            
            # if not all_picks:
            #     self.logger.error("No draft data was successfully retrieved")
            #     return pd.DataFrame()
                
            return pd.DataFrame(all_picks)
            
        except Exception as e:
            self.logger.error(f"Error in scrape_all_drafts: {e}")
            return pd.DataFrame()
    
    
    def get_team_current_stats(self, tricode: str, game_type: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get current season statistics for a specific team and game type"""
        try:
            season = self.get_current_season()
            url = f"{self.web_api_url}/club-stats/{tricode}/{season}/{game_type}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Retrieved current stats for {tricode}")

            return self._data_to_skaters_and_goalies_df(data, game_type,season,tricode)
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching stats for {tricode} game type {game_type}: {e}")
            return pd.DataFrame(), pd.DataFrame()


    async def _scrape_current_season_async(self, team_codes: Optional[Union[str, List[str]]] = None) -> Dict[str, pd.DataFrame]:
        """Scrape current season data for all game types
        return Dict has keys 'teams', 'skaters', 'goalies' with corresponding DataFrames
        """
        season = self.get_current_season()
        all_skaters = []
        all_goalies = []
        processed_teams = set()
        
            # Handle single team code input
        if isinstance(team_codes, str):
            team_codes = [team_codes]
        
        # Use provided team codes or fall back to all active teams
        teams_to_scrape = team_codes if team_codes is not None else self.active_team_codes
    
        # Validate team codes if provided
        if team_codes is not None:
            invalid_teams = [team for team in teams_to_scrape if team not in self.active_team_codes]
            if invalid_teams:
                raise ValueError(f"Invalid team code(s): {invalid_teams}")
    
        urls = [
            f"{self.web_api_url}/club-stats/{tricode}/{season}/{game_type}"
            for tricode in teams_to_scrape
            for game_type in self.game_types
        ]  
        responses = await self._fetch_all_data(urls)

        # Process responses in chunks corresponding to game types
        chunk_size = len(self.game_types)
        for i in range(0, len(responses), chunk_size):
            team_responses = responses[i:i + chunk_size]
            team_idx = i // chunk_size
            team = teams_to_scrape[team_idx]
            
            team_has_data = False
            
            for resp_idx, response in enumerate(team_responses):
                if response is not None:
                    game_type = self.game_types[resp_idx]
                    skaters_df, goalies_df = self._data_to_skaters_and_goalies_df(
                        response,
                        game_type=game_type,
                        tricode=team
                    )
                    
                    if not skaters_df.empty or not goalies_df.empty:
                        team_has_data = True
                        
                        if not skaters_df.empty:
                            all_skaters.append(skaters_df)
                        if not goalies_df.empty:
                            all_goalies.append(goalies_df)
            
            if team_has_data:
                processed_teams.add(team)
        
        # Create final dataframes
        current_season_data = {
            'teams': pd.DataFrame(processed_teams),
            'skaters': pd.concat(all_skaters, ignore_index=True) if all_skaters else pd.DataFrame(),
            'goalies': pd.concat(all_goalies, ignore_index=True) if all_goalies else pd.DataFrame()
        }
        
        # Add season information
        current_season = self.get_current_season()
        for df in current_season_data.values():
            if not df.empty:
                df['season'] = current_season
        
        return current_season_data
    
    async def scrape_current_season(
        self, 
        team_codes: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Synchronous wrapper for the async scraper"""
        loop = asyncio.get_event_loop()
        return await self._scrape_current_season_async(team_codes)

    async def scrape_team_gametypes(self, tricode: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """Scrape all gametypes (Seasons Reg/PO) for a team(tricode) or all active teams if no tricode is provided."""
        if tricode is None:
            tricodes = self.active_team_codes
        elif isinstance(tricode, str):
            tricodes = [tricode]
        else:
            tricodes = tricode

        urls = [f"{self.web_api_url}/club-stats-season/{code}" for code in tricodes]

        responses = await self._fetch_all_data(urls)

        all_data = []
        for response, code in zip(responses, tricodes):
            if response:
                for entry in response:
                    entry['triCode'] = code  # Add the team's tricode to each entry
                all_data.extend(response)

        return pd.DataFrame(all_data)
    
    async def async_scrape_all_seasons(self, active_only: bool = True):
        """Async version of scrape_all_seasons_by_gametype."""
        teams = self.get_all_teams()['triCode'].tolist()
        if active_only:
            teams = [team for team in teams if team in self.active_team_codes]

        # Fetch all gametypes in parallel
        self.logger.info(f"Fetching gametypes for {len(teams)} teams...")
        gametype_tasks = [self.scrape_team_gametypes(tricode) for tricode in teams]
        gametype_results = await asyncio.gather(*gametype_tasks, return_exceptions=True)

        # Build URL list with metadata
        all_urls = []
        
        for tricode, result in zip(teams, gametype_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error gathering URLs for team {tricode}: {result}")
                continue

            try:
                for entry in result.to_dict(orient='records'):
                    season = entry['season']
                    for game_type in entry['gameTypes']: 
                        url = f"{self.web_api_url}/club-stats/{tricode}/{season}/{game_type}"
                        url_data = {
                            'url': url,
                            'tricode': tricode,
                            'season': season,
                            'gameType': game_type
                        }
                        all_urls.append(url_data)
                self.logger.info(f"Gathered URLs for team {tricode}")
            except Exception as e:
                self.logger.error(f"Error processing gametypes for team {tricode}: {e}")
                continue

        # Extract just the URLs for the API call
        urls = [item['url'] for item in all_urls]
        
        self.logger.info(f"Fetching data for {len(urls)} team-season-gametype combinations...")
        
        # Fetch all data asynchronously
        responses = await self._fetch_all_data(urls)
        
        # Combine responses with metadata
        processed_responses = []
        for response, metadata in zip(responses, all_urls):
            if response is None:
                self.logger.warning(
                    f"Failed to fetch data for {metadata['tricode']} "
                    f"{metadata['season']} {metadata['gameType']}"
                )
                continue
                
            # Merge the API response with the metadata
            if isinstance(response, dict):
                response.update({
                    'teamTricode': metadata['tricode'],
                    'season': metadata['season'],
                    'gameType': metadata['gameType']
                })
                processed_responses.append(response)
            else:
                # Handle non-dictionary responses if needed
                processed_response = {
                    'data': response,
                    'teamTricode': metadata['tricode'],
                    'season': metadata['season'],
                    'gameType': metadata['gameType']
                }
                processed_responses.append(processed_response)
        
        self.logger.info(f"Successfully processed {len(processed_responses)}/{len(urls)} requests")
        return processed_responses

    def scrape_all_seasons_by_gametype(self, active_only: bool = True):
        """
        Scrape all team data, including gametypes and seasons.
        Optionally restrict to active teams only.
        """
        try:
            # Get the current event loop if it exists
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running, create and run a new one
            return asyncio.run(self.async_scrape_all_seasons(active_only))
        else:
            # Return coroutine directly to allow 'await'
            return self.async_scrape_all_seasons(active_only)
        
    async def process_all_teams(self, active_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes all team data from `scraper.scrape_all_seasons_by_gametype()` 
        and merges all skater and goalie DataFrames.
        
        Args:
            active_only: If True, only process active teams. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Combined Skaters and Goalies DataFrames.
        """
        # Step 1: Fetch all team data
        data = await self.scrape_all_seasons_by_gametype(active_only=active_only)

        # Step 2: Initialize empty lists for skater and goalie DataFrames
        all_skaters = []
        all_goalies = []

        # Step 3: Process each team's data
        for team_data in data:  # Assuming `data` is a list of team dictionaries
            team_skater_df, team_goalie_df = self._data_to_skaters_and_goalies_df(
                team_data,
                game_type=team_data.get("gameType"),  # Optional fields if present
                season=team_data.get("season"),
                tricode=team_data.get("teamTricode"),
            )
            if not team_skater_df.empty:
                all_skaters.append(team_skater_df)
            if not team_goalie_df.empty:
                all_goalies.append(team_goalie_df)

        # Step 4: Concatenate all skater and goalie DataFrames
        combined_skaters_df = pd.concat(all_skaters, ignore_index=True) if all_skaters else pd.DataFrame()
        combined_goalies_df = pd.concat(all_goalies, ignore_index=True) if all_goalies else pd.DataFrame()

        return combined_skaters_df, combined_goalies_df

    # Columns read by insert_players_from_staging_with_logging().
    # Landing payloads omit these for unnumbered camp/prospect players.
    PLAYER_STAGING_COLUMNS = (
        "playerId",
        "isActive",
        "currentTeamId",
        "currentTeamAbbrev",
        "fullTeamName.default",
        "firstName.default",
        "lastName.default",
        "sweaterNumber",
        "position",
        "headshot",
        "heroImage",
        "heightInInches",
        "heightInCentimeters",
        "weightInPounds",
        "weightInKilograms",
        "birthDate",
        "birthCity.default",
        "birthStateProvince.default",
        "birthCountry",
        "shootsCatches",
        "playerSlug",
        "inTop100AllTime",
        "inHHOF",
        "draftDetails.year",
        "draftDetails.teamAbbrev",
        "draftDetails.round",
        "draftDetails.pickInRound",
        "draftDetails.overallPick",
    )

    SEASON_LOCALE_COLUMNS = (
        "teamCommonName.default",
        "teamCommonName.cs",
        "teamCommonName.de",
        "teamCommonName.es",
        "teamCommonName.fi",
        "teamCommonName.fr",
        "teamCommonName.sk",
        "teamCommonName.sv",
        "teamName.default",
        "teamName.cs",
        "teamName.de",
        "teamName.fi",
        "teamName.fr",
        "teamName.sk",
        "teamName.sv",
        "teamPlaceNameWithPreposition.default",
        "teamPlaceNameWithPreposition.cs",
        "teamPlaceNameWithPreposition.es",
        "teamPlaceNameWithPreposition.fi",
        "teamPlaceNameWithPreposition.fr",
        "teamPlaceNameWithPreposition.sk",
        "teamPlaceNameWithPreposition.sv",
    )

    SEASON_SKATER_STAGING_COLUMNS = (
        "playerId",
        "assists",
        "gameTypeId",
        "gamesPlayed",
        "goals",
        "leagueAbbrev",
        "pim",
        "plusMinus",
        "points",
        "season",
        "sequence",
        "faceoffWinningPctg",
        "shootingPctg",
        "shots",
        "powerPlayGoals",
        "shorthandedGoals",
        "gameWinningGoals",
        "avgToi",
        "otGoals",
        "powerPlayPoints",
        "shorthandedPoints",
    ) + SEASON_LOCALE_COLUMNS

    SEASON_GOALIE_STAGING_COLUMNS = (
        "playerId",
        "gameTypeId",
        "gamesPlayed",
        "goalsAgainst",
        "goalsAgainstAvg",
        "leagueAbbrev",
        "losses",
        "season",
        "sequence",
        "shutouts",
        "ties",
        "timeOnIce",
        "wins",
        "assists",
        "gamesStarted",
        "goals",
        "pim",
        "savePctg",
        "shotsAgainst",
        "otLosses",
    ) + SEASON_LOCALE_COLUMNS

    @staticmethod
    def _ensure_dataframe_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.DataFrame:
        """Add nullable columns expected by staging sync procedures."""
        if df is None:
            return df
        for column in columns:
            if column not in df.columns:
                df[column] = None
        return df
    
    def scrape_player(self, player_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        url = f"{self.web_api_url}/player/{player_id}/landing"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        player_df = pd.json_normalize(data)
        # Process seasonTotals data
        seasons_df = self._process_season_totals(player_df, player_id)
        # Process awards data
        awards_df = (self._process_awards(player_df, player_id) 
                    if 'awards' in player_df.columns 
                    else pd.DataFrame())        
        cols_to_drop = [
        'badges', 'last5Games', 'seasonTotals', 
        'currentTeamRoster', 'awards', 'shopLink', 
        'twitterLink', 'watchLink'
        ]
        existing_cols = [col for col in cols_to_drop if col in player_df.columns]

        player_df.drop(columns=existing_cols, inplace=True)
        player_df = NHLScraper._ensure_dataframe_columns(
            player_df, NHLScraper.PLAYER_STAGING_COLUMNS
        )
        return player_df, seasons_df, awards_df
    
    async def scrape_all_players(self, player_ids: List[str], engine,batch_size: int = 100) -> None:
        # Deduplicate player_ids to prevent duplicate processing
        player_ids = list(set(player_ids))

        # Create URLs for all players
        all_players = []
        all_skater_seasons = []
        all_goalie_seasons = []
        all_awards = []

        try:
            urls = [f"{self.web_api_url}/player/{player_id}/landing" for player_id in player_ids]
            # Process in batches
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_ids = player_ids[i:i + batch_size]
                
                # Fetch batch of data
                responses = await self._fetch_all_data(batch_urls)
                
                # Process responses
                for response, player_id in zip(responses, batch_ids):
                    if response:
                            player_df = pd.json_normalize(response)
                            seasons_df = self._process_season_totals(player_df, player_id)
                            
                            # Check position to split seasons
                            position = player_df['position'].iloc[0] if 'position' in player_df.columns else None
                            if position == 'G':
                                all_goalie_seasons.append(seasons_df)
                            else:
                                all_skater_seasons.append(seasons_df)

                            awards_df = (self._process_awards(player_df, player_id) 
                                    if 'awards' in player_df.columns 
                                    else pd.DataFrame())

                            cols_to_drop = ['badges', 'last5Games', 'seasonTotals', 
                                        'currentTeamRoster', 'awards', 'shopLink', 
                                        'twitterLink', 'watchLink']
                            existing_cols = [col for col in cols_to_drop if col in player_df.columns]
                            player_df.drop(columns=existing_cols, inplace=True)
                            
                            all_players.append(player_df)
                            if awards_df is not None and not awards_df.empty:
                                all_awards.append(awards_df)

                current_batch = i // batch_size + 1
                total_batches = (len(urls) + batch_size - 1) // batch_size
                self.logger.info(f"Processed batch {current_batch}/{total_batches} ({len(batch_urls)} players)")

            with engine.begin() as conn:
                if all_players:
                    try:
                        player_staging_df = NHLScraper._ensure_dataframe_columns(
                            pd.concat(all_players, ignore_index=True),
                            NHLScraper.PLAYER_STAGING_COLUMNS,
                        )
                        player_staging_df.to_sql('player', conn, if_exists='replace', index=False, schema='staging1')
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.player: {e}")
                        raise
                if all_skater_seasons:
                    try:
                        skater_seasons_df = NHLScraper._ensure_dataframe_columns(
                            pd.concat(all_skater_seasons, ignore_index=True),
                            NHLScraper.SEASON_SKATER_STAGING_COLUMNS,
                        )
                        skater_seasons_df.to_sql('season_skater', conn, if_exists='replace', index=False, schema='staging1')
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.season_skater: {e}")
                        raise
                if all_goalie_seasons:
                    try:
                        goalie_seasons_df = NHLScraper._ensure_dataframe_columns(
                            pd.concat(all_goalie_seasons, ignore_index=True),
                            NHLScraper.SEASON_GOALIE_STAGING_COLUMNS,
                        )
                        goalie_seasons_df.to_sql('season_goalie', conn, if_exists='replace', index=False, schema='staging1')
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.season_goalie: {e}")
                        raise
                if all_awards:
                    try:
                        awards_staging_df = pd.concat(all_awards, ignore_index=True)
                        if not awards_staging_df.empty and "playerId" in awards_staging_df.columns:
                            awards_staging_df.to_sql('award', conn, if_exists='replace', index=False, schema='staging1')
                        else:
                            self.logger.info("No award rows to stage; leaving staging1.award unchanged")
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.award: {e}")
                        raise
                # commit is automatic with engine.begin()
        except Exception as e:
            self.logger.error(f"Error in scrape_all_players: {e}")
            raise
            

    async def _get_all_player_columns(self, player_ids: List[str], batch_size: int = 50):
        urls = [f"{self.web_api_url}/player/{player_id}/landing" for player_id in player_ids]
        all_columns = set()
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            batch_ids = player_ids[i:i + batch_size]
            
            responses = await self._fetch_all_data(batch_urls)
            
            for response, player_id in zip(responses, batch_ids):
                if response:
                    try:
                        player_df = pd.json_normalize(response)
                        cols_to_drop = ['badges', 'last5Games', 'seasonTotals', 
                                    'currentTeamRoster', 'awards', 'shopLink', 
                                    'twitterLink', 'watchLink']
                        existing_cols = [col for col in cols_to_drop if col in player_df.columns]
                        player_df.drop(columns=existing_cols, inplace=True)
                        
                        all_columns.update(player_df.columns)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing player {player_id}: {e}")
            
            self.logger.info(f"Processed batch {i//batch_size + 1}")
        
        return sorted(list(all_columns))
    
    def _process_season_totals(self, player_df: pd.DataFrame, player_id: str) -> pd.DataFrame:
        """Processes season totals data."""
        seasons_df = pd.json_normalize(player_df['seasonTotals'].iloc[0])
        seasons_df.insert(0, 'playerId', player_id)
        return seasons_df

    def _process_awards(self, player_df: pd.DataFrame, player_id: str) -> pd.DataFrame:
        """Processes and expands awards data."""
        awards_table = pd.json_normalize(player_df['awards'].iloc[0])
        
        # Use list comprehension for better performance
        award_rows = [
            {
                'playerId': player_id,
                'trophy_default': row['trophy.default'],
                'trophy_fr': row.get('trophy.fr'),
                **season
            }
            for _, row in awards_table.iterrows()
            for season in row['seasons']
        ]
        
        return pd.DataFrame(award_rows)
    
    def _data_to_skaters_and_goalies_df(self, data, game_type=None, season=None, tricode=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Processes a Dict[str, DF] skater/player dict (output from `scrape_current_season` etc.) into skaters and goalies DataFrames
        """
        skaters_df = pd.DataFrame(data.get('skaters', []))
        if not skaters_df.empty:
            skaters_df['firstName'] = skaters_df['firstName'].apply(lambda x: x.get('default', ''))
            skaters_df['lastName'] = skaters_df['lastName'].apply(lambda x: x.get('default', ''))
            skaters_df['fullName'] = skaters_df['firstName'] + ' ' + skaters_df['lastName']
            if game_type:
                skaters_df['gameType'] = game_type
            if season:
                skaters_df['season'] = season
            if tricode:
                skaters_df['triCode'] = tricode
        
        # Process goalies
        goalies_df = pd.DataFrame(data.get('goalies', []))
        if not goalies_df.empty:
            goalies_df['firstName'] = goalies_df['firstName'].apply(lambda x: x.get('default', ''))
            goalies_df['lastName'] = goalies_df['lastName'].apply(lambda x: x.get('default', ''))
            goalies_df['fullName'] = goalies_df['firstName'] + ' ' + goalies_df['lastName']
            if game_type:
                goalies_df['gameType'] = game_type
            if season:
                goalies_df['season'] = season
            if tricode:
                goalies_df['team'] = tricode
        
        return skaters_df, goalies_df
    
    async def scrape_all_rosters(self, team_codes: Optional[List[str]] = None, delay: float = 0.7) -> pd.DataFrame:
        """
        Scrape current rosters for all teams or specified teams.
        
        Args:
            team_codes: Optional list of team abbreviations. If None, scrapes all active teams.
            delay: Delay between requests in seconds to avoid rate limiting.
            
        Returns:
            DataFrame with all players from all team rosters, including columns:
            - teamAbbreviation: Team code (e.g., 'TOR', 'NYR')
            - position: 'forwards', 'defensemen', or 'goalies'
            - playerId: Player's unique ID
            - sweaterNumber: Jersey number
            - firstName: Player's first name
            - lastName: Player's last name
        """
        teams_to_scrape = team_codes if team_codes is not None else self.active_team_codes
        
        # Validate team codes if provided
        if team_codes is not None:
            invalid_teams = [team for team in teams_to_scrape if team not in self.active_team_codes]
            if invalid_teams:
                raise ValueError(f"Invalid team code(s): {invalid_teams}")
        
        all_players = []
        
        async with aiohttp.ClientSession() as session:
            for team in teams_to_scrape:
                try:
                    url = f"{self.web_api_url}/roster/{team}/current"
                    # self.logger.info(f"Fetching roster for {team}...")
                    
                    async with session.get(url) as res:
                        if res.status != 200:
                            self.logger.error(f"Failed to fetch roster for {team}: {res.status}")
                            continue
                        
                        data = await res.json()
                        self.logger.info(f"Successfully fetched roster for {team}")
                        
                        # Process each position group
                        for position in ['forwards', 'defensemen', 'goalies']:
                            players = data.get(position, [])
                            for player in players:
                                player_data = {
                                    'teamAbbreviation': team,
                                    'positionGroup': position,
                                    'playerId': player.get('id'),
                                    'headshot': player.get('headshot'),
                                    'firstName': player.get('firstName', {}).get('default', ''),
                                    'lastName': player.get('lastName', {}).get('default', ''),
                                    'sweaterNumber': player.get('sweaterNumber'),
                                    'positionCode': player.get('positionCode'),
                                    'shootsCatches': player.get('shootsCatches'),
                                    'heightInInches': player.get('heightInInches'),
                                    'weightInPounds': player.get('weightInPounds'),
                                    'heightInCentimeters': player.get('heightInCentimeters'),
                                    'weightInKilograms': player.get('weightInKilograms'),
                                    'birthDate': player.get('birthDate'),
                                    'birthCity': player.get('birthCity', {}).get('default', ''),
                                    'birthCountry': player.get('birthCountry'),
                                    'birthStateProvince': player.get('birthStateProvince', {}).get('default', ''),
                                }
                                all_players.append(player_data)
                    
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Error fetching roster for {team}: {e}")
                    continue
        
        # Create DataFrame and sort by team and full name
        df = pd.DataFrame(all_players)
        if not df.empty:
            df['fullName'] = df['firstName'] + ' ' + df['lastName']
            df = df.sort_values(['teamAbbreviation', 'positionGroup', 'fullName'])
            df = df.reset_index(drop=True)
        
        return df

    async def _fetch_data(self, session, url):
        """Fetch data from a single URL."""
        try:
            async with session.get(url) as res:
                if res.status == 200:
                    return await res.json()
                else:
                    self.logger.error(f"Error {res.status} for URL: {url}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    async def _fetch_all_data(
        self,
        urls: List[str],
        batch_size: int = 10,
        delay_between_batches: float = 0.5,
    ) -> List[dict]:
        """Fetch data from multiple URLs concurrently.

        Kept simple and delegates per-request retry/backoff to `_fetch_data`.
        Returns a list with the responses (or None for failed requests) in the same order as `urls`.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_data(session, url) for url in urls]
            return await asyncio.gather(*tasks)

    def get_team_schedule(self, team: str, season: str = "now") -> Dict:
        """
        Get schedule for a specific team.

        Args:
            team: Team tricode (e.g., 'TOR', 'NYR')
            season: Season string (e.g., '20232024') or 'now' for current season

        Returns:
            Dict containing schedule data with 'games' key
        """
        if season == "now":
            season = self.get_current_season()

        url = f"{self.web_api_url}/club-schedule-season/{team}/{season}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    async def get_gamecenter_staging_data(
        self,
        game_ids: List[int],
        batch_size: int = 50,
        delay_between_batches: float = 0.1,
    ) -> pd.DataFrame:
        """Fetch gamecenter play-by-play data and return raw rows for staging1.gamecenter."""
        game_ids = [int(gid) for gid in dict.fromkeys(game_ids) if gid is not None]
        total = len(game_ids)
        rows = []
        failed_game_ids = []
        games_with_plays = 0

        if not game_ids:
            return pd.DataFrame()

        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=batch_size)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for start in range(0, total, batch_size):
                batch = game_ids[start:start + batch_size]
                batch_number = (start // batch_size) + 1
                total_batches = (total + batch_size - 1) // batch_size
                batch_started_at = datetime.now()

                urls = [
                    f"{self.web_api_url}/gamecenter/{game_id}/play-by-play"
                    for game_id in batch
                ]
                responses = await asyncio.gather(
                    *(self._fetch_data(session, url) for url in urls),
                    return_exceptions=True,
                )

                batch_rows = 0
                batch_successes = 0
                batch_failures = []
                for game_id, response in zip(batch, responses):
                    if isinstance(response, Exception) or response is None:
                        batch_failures.append(game_id)
                        continue

                    plays = response.get("plays", []) if isinstance(response, dict) else []
                    if not plays:
                        batch_successes += 1
                        continue

                    games_with_plays += 1
                    batch_successes += 1
                    for play in plays:
                        if not isinstance(play, dict):
                            continue
                        event_id = play.get("eventId")
                        if event_id is not None:
                            rows.append({
                                "game_id": response.get("id") or game_id,
                                "event_id": event_id,
                                "game_payload": response,
                                "raw_play": play,
                            })
                            batch_rows += 1

                failed_game_ids.extend(batch_failures)
                elapsed = (datetime.now() - batch_started_at).total_seconds()
                processed = min(start + len(batch), total)
                self.logger.info(
                    "Gamecenter batch %s/%s: processed %s/%s games, %s successes, %s failures, %s play rows in %.1fs",
                    batch_number,
                    total_batches,
                    processed,
                    total,
                    batch_successes,
                    len(batch_failures),
                    batch_rows,
                    elapsed,
                )

                if batch_failures:
                    self.logger.warning(
                        "Gamecenter batch %s failed game ids: %s",
                        batch_number,
                        batch_failures,
                    )

                if delay_between_batches and start + batch_size < total:
                    await asyncio.sleep(delay_between_batches)

        self.logger.info(
            "Completed gamecenter fetch: %s games attempted, %s games with plays, %s failed games, %s total play rows",
            total,
            games_with_plays,
            len(failed_game_ids),
            len(rows),
        )
        if failed_game_ids:
            self.logger.warning("Gamecenter failed game ids: %s", failed_game_ids)

        return pd.DataFrame(rows)

    def scrape_all_games_team_method(self, season: str = "now", delay: float = 0.7) -> List[Dict]:
        """
        Get all teams' schedules and combine them (with deduplication).

        Args:
            season: Season string (e.g., '20232024') or 'now' for current season

        Returns:
            List of unique game dictionaries
        """
        all_games = []
        seen_game_ids = set()

        for i, team in enumerate(self.active_team_codes, 1):
            self.logger.info(f"Fetching schedule for {team} ({i}/{len(self.active_team_codes)})...")
            try:
                schedule = self.get_team_schedule(team, season)

                # Extract games from the schedule
                if "games" in schedule:
                    for game in schedule["games"]:
                        game_id = game.get("id")
                        # Avoid duplicates
                        if game_id and game_id not in seen_game_ids:
                            seen_game_ids.add(game_id)
                            all_games.append(game)
            except Exception as e:
                self.logger.error(f"Error fetching schedule for {team}: {e}")
                continue
            finally:
                # small delay between requests to avoid triggering rate limits
                try:
                    sleep(delay)
                except Exception:
                    pass

        self.logger.info(f"Fetched {len(all_games)} unique games from {len(self.active_team_codes)} teams")
        return all_games

    def _games_to_dataframe(self, games: List[Dict]) -> pd.DataFrame:
        """Flatten NHL schedule/gamecenter game payloads into the games staging shape."""
        if not games:
            return pd.DataFrame()

        df = pd.json_normalize(games, sep='_')

        if 'tvBroadcasts' in df.columns:
            df['tvBroadcasts'] = df['tvBroadcasts'].apply(
                lambda broadcasts: ', '.join(
                    broadcast.get('network', '')
                    for broadcast in broadcasts
                    if isinstance(broadcast, dict)
                ) if isinstance(broadcasts, list) else ''
            )

        return df

    @staticmethod
    def _default_text(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            return value.get("default")
        return value

    @staticmethod
    def _full_name(player: Optional[Dict]) -> Optional[str]:
        if not isinstance(player, dict):
            return None
        first = NHLScraper._default_text(player.get("firstName"))
        last = NHLScraper._default_text(player.get("lastName"))
        return " ".join(part for part in (first, last) if part) or None

    def _game_summary_dataframes(self, landing_payloads: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Extract compact goal, penalty, and three-star rows from landing summary payloads."""
        goal_rows = []
        penalty_rows = []
        star_rows = []

        for payload in landing_payloads:
            if not isinstance(payload, dict):
                continue

            game_id = payload.get("id")
            season = payload.get("season")
            game_type = payload.get("gameType")
            game_date = payload.get("gameDate")
            away_team = (payload.get("awayTeam") or {}).get("abbrev")
            home_team = (payload.get("homeTeam") or {}).get("abbrev")
            summary = payload.get("summary") or {}

            for period in summary.get("scoring") or []:
                period_descriptor = period.get("periodDescriptor") or {}
                for goal in period.get("goals") or []:
                    assists = goal.get("assists") or []
                    assist1 = assists[0] if len(assists) > 0 else {}
                    assist2 = assists[1] if len(assists) > 1 else {}
                    goal_rows.append({
                        "game_id": game_id,
                        "event_id": goal.get("eventId"),
                        "season": season,
                        "game_type": game_type,
                        "game_date": game_date,
                        "away_team_abbrev": away_team,
                        "home_team_abbrev": home_team,
                        "period_number": period_descriptor.get("number"),
                        "period_type": period_descriptor.get("periodType"),
                        "time_in_period": goal.get("timeInPeriod"),
                        "team_abbrev": self._default_text(goal.get("teamAbbrev")),
                        "is_home": goal.get("isHome"),
                        "strength": goal.get("strength"),
                        "situation_code": goal.get("situationCode"),
                        "scoring_player_id": goal.get("playerId"),
                        "scoring_player_name": self._default_text(goal.get("name")),
                        "assist1_player_id": assist1.get("playerId"),
                        "assist1_player_name": self._default_text(assist1.get("name")),
                        "assist2_player_id": assist2.get("playerId"),
                        "assist2_player_name": self._default_text(assist2.get("name")),
                        "away_score": goal.get("awayScore"),
                        "home_score": goal.get("homeScore"),
                    })

            for period in summary.get("penalties") or []:
                period_descriptor = period.get("periodDescriptor") or {}
                for idx, penalty in enumerate(period.get("penalties") or [], 1):
                    committed_by = penalty.get("committedByPlayer") or {}
                    drawn_by = penalty.get("drawnBy") or {}
                    penalty_rows.append({
                        "game_id": game_id,
                        "penalty_index": idx,
                        "season": season,
                        "game_type": game_type,
                        "game_date": game_date,
                        "away_team_abbrev": away_team,
                        "home_team_abbrev": home_team,
                        "period_number": period_descriptor.get("number"),
                        "period_type": period_descriptor.get("periodType"),
                        "time_in_period": penalty.get("timeInPeriod"),
                        "team_abbrev": self._default_text(penalty.get("teamAbbrev")),
                        "penalty_type": penalty.get("type"),
                        "duration": penalty.get("duration"),
                        "desc_key": penalty.get("descKey"),
                        "committed_by_player_name": self._full_name(committed_by),
                        "committed_by_sweater_number": committed_by.get("sweaterNumber"),
                        "drawn_by_player_name": self._full_name(drawn_by),
                        "drawn_by_sweater_number": drawn_by.get("sweaterNumber"),
                        "served_by_name": self._default_text(penalty.get("servedBy")),
                    })

            for star in summary.get("threeStars") or []:
                star_rows.append({
                    "game_id": game_id,
                    "star": star.get("star"),
                    "season": season,
                    "game_type": game_type,
                    "game_date": game_date,
                    "player_id": star.get("playerId"),
                    "player_name": self._default_text(star.get("name")),
                    "team_abbrev": star.get("teamAbbrev"),
                    "sweater_number": star.get("sweaterNo"),
                    "position": star.get("position"),
                    "goals": star.get("goals"),
                    "assists": star.get("assists"),
                    "points": star.get("points"),
                })

        goal_columns = [
            "game_id", "event_id", "season", "game_type", "game_date",
            "away_team_abbrev", "home_team_abbrev", "period_number", "period_type",
            "time_in_period", "team_abbrev", "is_home", "strength", "situation_code",
            "scoring_player_id", "scoring_player_name", "assist1_player_id",
            "assist1_player_name", "assist2_player_id", "assist2_player_name",
            "away_score", "home_score",
        ]
        penalty_columns = [
            "game_id", "penalty_index", "season", "game_type", "game_date",
            "away_team_abbrev", "home_team_abbrev", "period_number", "period_type",
            "time_in_period", "team_abbrev", "penalty_type", "duration", "desc_key",
            "committed_by_player_name", "committed_by_sweater_number",
            "drawn_by_player_name", "drawn_by_sweater_number", "served_by_name",
        ]
        star_columns = [
            "game_id", "star", "season", "game_type", "game_date", "player_id",
            "player_name", "team_abbrev", "sweater_number", "position", "goals",
            "assists", "points",
        ]

        return {
            "game_goals": pd.DataFrame(goal_rows, columns=goal_columns),
            "game_penalties": pd.DataFrame(penalty_rows, columns=penalty_columns),
            "game_three_stars": pd.DataFrame(star_rows, columns=star_columns),
        }

    async def _fetch_game_landing_payloads(
        self,
        game_ids: List[int],
        batch_size: int = 100,
        delay_between_batches: float = 0.1,
    ) -> List[Dict]:
        """Fetch /gamecenter/{game_id}/landing payloads for richer game-level columns."""
        game_ids = [int(game_id) for game_id in dict.fromkeys(game_ids) if game_id is not None]
        if not game_ids:
            return []

        payloads = []
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=batch_size)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for start in range(0, len(game_ids), batch_size):
                batch = game_ids[start:start + batch_size]
                urls = [
                    f"{self.web_api_url}/gamecenter/{game_id}/landing"
                    for game_id in batch
                ]
                responses = await asyncio.gather(
                    *(self._fetch_data(session, url) for url in urls),
                    return_exceptions=True,
                )

                for game_id, response in zip(batch, responses):
                    if isinstance(response, Exception) or response is None:
                        self.logger.warning("Failed to fetch landing payload for game %s", game_id)
                        continue
                    payloads.append(response)

                self.logger.info(
                    "Fetched landing payload batch %s/%s (%s/%s games)",
                    (start // batch_size) + 1,
                    (len(game_ids) + batch_size - 1) // batch_size,
                    min(start + len(batch), len(game_ids)),
                    len(game_ids),
                )

                if delay_between_batches and start + batch_size < len(game_ids):
                    await asyncio.sleep(delay_between_batches)

        return payloads

    def _fetch_game_landing_payloads_sync(
        self,
        game_ids: List[int],
        batch_size: int = 100,
        delay_between_batches: float = 0.1,
    ) -> List[Dict]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._fetch_game_landing_payloads(
                    game_ids,
                    batch_size=batch_size,
                    delay_between_batches=delay_between_batches,
                )
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    self._fetch_game_landing_payloads(
                        game_ids,
                        batch_size=batch_size,
                        delay_between_batches=delay_between_batches,
                    )
                )
            )
            return future.result()

    def scrape_all_games_to_dataframe(
        self,
        season: str = "now",
        enrich_from_landing: bool = True,
        landing_batch_size: int = 100,
        landing_delay_between_batches: float = 0.1,
    ) -> pd.DataFrame:
        """
        Scrape all games and return as a DataFrame.

        Args:
            season: Season string (e.g., '20232024') or 'now' for current season
            enrich_from_landing: If True, fetch /gamecenter/{game_id}/landing
                for richer game-level fields such as shots, logos, venue location,
                clock, and playoff/OT flags.

        Returns:
            DataFrame with game data, flattened for SQL compatibility
        """
        games = self.scrape_all_games_team_method(season)
        if not games:
            return pd.DataFrame()

        if enrich_from_landing:
            game_ids = [game.get("id") for game in games if game.get("id") is not None]
            landing_payloads = self._fetch_game_landing_payloads_sync(
                game_ids,
                batch_size=landing_batch_size,
                delay_between_batches=landing_delay_between_batches,
            )
            if landing_payloads:
                self.logger.info(
                    "Using %s landing payloads for games dataframe enrichment",
                    len(landing_payloads),
                )
                return self._ensure_games_staging_columns(self._games_to_dataframe(landing_payloads))

        return self._ensure_games_staging_columns(self._games_to_dataframe(games))

    def scrape_all_games_dataframes(
        self,
        season: str = "now",
        landing_batch_size: int = 100,
        landing_delay_between_batches: float = 0.1,
        dates: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Scrape games plus compact landing summary tables using one landing fetch per game.

        Args:
            season: Season string (e.g., '20232024') or 'now' for current season.
            dates: Optional list of 'YYYY-MM-DD' schedule dates. When provided, only
                those days are scraped instead of every team schedule for the season.
        """
        if dates:
            games = self.get_games_for_dates(dates)
        else:
            games = self.scrape_all_games_team_method(season)
        return self._games_dataframes_from_games(
            games,
            landing_batch_size=landing_batch_size,
            landing_delay_between_batches=landing_delay_between_batches,
        )

    def _games_dataframes_from_games(
        self,
        games: List[Dict],
        landing_batch_size: int = 100,
        landing_delay_between_batches: float = 0.1,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch landing payloads for a game list and return games plus summary tables."""
        if not games:
            return {
                "games": pd.DataFrame(),
                "game_goals": pd.DataFrame(),
                "game_penalties": pd.DataFrame(),
                "game_three_stars": pd.DataFrame(),
            }

        game_ids = [game.get("id") for game in games if game.get("id") is not None]
        landing_payloads = self._fetch_game_landing_payloads_sync(
            game_ids,
            batch_size=landing_batch_size,
            delay_between_batches=landing_delay_between_batches,
        )

        source_games = landing_payloads if landing_payloads else games
        dataframes = self._game_summary_dataframes(landing_payloads)
        dataframes["games"] = self._ensure_games_staging_columns(self._games_to_dataframe(source_games))
        return dataframes

    def get_nhl_calendar_date(self, date: Optional[str] = None):
        """Return an NHL schedule calendar date, using America/New_York when omitted."""
        if date:
            return datetime.strptime(date, "%Y-%m-%d").date()
        return datetime.now(NHL_SCHEDULE_TZ).date()

    def get_schedule_window_dates(
        self,
        days: int = 2,
        end_date: Optional[str] = None,
        lookahead_days: int = 0,
    ) -> List[str]:
        """Return NHL schedule dates around today (or end_date).

        days=2 and lookahead_days=1 is tomorrow, today, and yesterday
        in America/New_York.
        """
        if days < 1:
            raise ValueError("days must be at least 1")
        if lookahead_days < 0:
            raise ValueError("lookahead_days must be at least 0")

        end = self.get_nhl_calendar_date(end_date)
        return [
            (end - timedelta(days=offset)).strftime("%Y-%m-%d")
            for offset in range(-lookahead_days, days)
        ]

    def get_games_for_dates(self, dates: Optional[List[str]] = None) -> List[Dict]:
        """Get unique games for one or more NHL schedule dates."""
        if not dates:
            dates = [self.get_nhl_calendar_date().strftime("%Y-%m-%d")]

        games = []
        seen_game_ids = set()
        for date in dates:
            for game in self.get_todays_games(date):
                game_id = game.get("id")
                game["gameDate"] = game.get("gameDate") or date
                game["scheduleDate"] = game.get("scheduleDate") or date
                if game_id and game_id in seen_game_ids:
                    continue
                if game_id:
                    seen_game_ids.add(game_id)
                games.append(game)
        return games

    def get_teams_for_dates(self, dates: Optional[List[str]] = None) -> List[str]:
        """Get unique team tricodes playing on one or more NHL schedule dates."""
        teams = set()
        for game in self.get_games_for_dates(dates):
            away = game.get("awayTeam", {}).get("abbrev")
            home = game.get("homeTeam", {}).get("abbrev")
            if away:
                teams.add(away)
            if home:
                teams.add(home)
        return sorted(teams)

    def get_todays_games(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get all games scheduled for today (or a specific date).

        Args:
            date: Optional date string in 'YYYY-MM-DD' format. Defaults to today
                in America/New_York.

        Returns:
            List of game dictionaries for the specified date
        """
        if date is None:
            date = self.get_nhl_calendar_date().strftime('%Y-%m-%d')

        url = f"{self.web_api_url}/schedule/{date}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        games = []
        for game_week in data.get('gameWeek', []):
            if game_week.get('date') == date:
                for game in game_week.get('games', []):
                    game['gameDate'] = game_week.get('date')  # Add date from parent
                games.extend(game_week.get('games', []))
                break

        return games

    def get_schedule_now_games(self) -> List[Dict]:
        """
        Get the current schedule day from /schedule/now.

        The NHL API redirects /schedule/now to the current schedule date and returns
        a gameWeek array. This method uses the first gameWeek entry as the current
        schedule day and attaches that date to every returned game.
        """
        url = f"{self.web_api_url}/schedule/now"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        game_week = data.get("gameWeek", [])
        if not game_week:
            return []

        current_day = game_week[0]
        schedule_date = current_day.get("date")
        games = current_day.get("games", [])
        for game in games:
            game["gameDate"] = game.get("gameDate") or schedule_date
            game["scheduleDate"] = schedule_date

        return games

    def get_schedule_now_teams(self) -> List[str]:
        """Get unique team tricodes from the current /schedule/now schedule day."""
        teams = set()
        for game in self.get_schedule_now_games():
            away = game.get("awayTeam", {}).get("abbrev")
            home = game.get("homeTeam", {}).get("abbrev")
            if away:
                teams.add(away)
            if home:
                teams.add(home)

        return sorted(teams)

    def get_schedule_now_games_dataframe(self) -> pd.DataFrame:
        """Return /schedule/now games in the same staging shape used by games ETL."""
        games = self.get_schedule_now_games()
        if not games:
            return pd.DataFrame()

        df = self._games_to_dataframe(games)

        expected_columns = [
            "id",
            "season",
            "gameType",
            "gameDate",
            "scheduleDate",
            "gameState",
            "gameScheduleState",
            "startTimeUTC",
            "venueTimezone",
            "easternUTCOffset",
            "venueUTCOffset",
            "neutralSite",
            "venue_default",
            "venueLocation_default",
            "tvBroadcasts",
            "limitedScoring",
            "shootoutInUse",
            "regPeriods",
            "otInUse",
            "tiesInUse",
            "awayTeam_id",
            "awayTeam_abbrev",
            "awayTeam_commonName_default",
            "awayTeam_placeName_default",
            "awayTeam_score",
            "awayTeam_sog",
            "awayTeam_logo",
            "awayTeam_darkLogo",
            "homeTeam_id",
            "homeTeam_abbrev",
            "homeTeam_commonName_default",
            "homeTeam_placeName_default",
            "homeTeam_score",
            "homeTeam_sog",
            "homeTeam_logo",
            "homeTeam_darkLogo",
            "periodDescriptor_number",
            "periodDescriptor_periodType",
            "periodDescriptor_otPeriods",
            "periodDescriptor_maxRegulationPeriods",
            "gameOutcome_lastPeriodType",
            "winningGoalie_playerId",
            "winningGoalScorer_playerId",
            "clock_timeRemaining",
            "clock_secondsRemaining",
            "clock_running",
            "clock_inIntermission",
            "gameCenterLink",
        ]
        for column in expected_columns:
            if column not in df.columns:
                df[column] = None

        return df[expected_columns]

    def _ensure_games_staging_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add nullable landing-enrichment columns expected by sync_games_from_staging."""
        expected_columns = [
            "id",
            "season",
            "gameType",
            "gameDate",
            "gameState",
            "gameScheduleState",
            "startTimeUTC",
            "easternUTCOffset",
            "venueUTCOffset",
            "venueTimezone",
            "neutralSite",
            "venue_default",
            "venueLocation_default",
            "tvBroadcasts",
            "limitedScoring",
            "shootoutInUse",
            "regPeriods",
            "otInUse",
            "tiesInUse",
            "awayTeam_id",
            "awayTeam_abbrev",
            "awayTeam_commonName_default",
            "awayTeam_placeName_default",
            "awayTeam_score",
            "awayTeam_sog",
            "awayTeam_logo",
            "awayTeam_darkLogo",
            "homeTeam_id",
            "homeTeam_abbrev",
            "homeTeam_commonName_default",
            "homeTeam_placeName_default",
            "homeTeam_score",
            "homeTeam_sog",
            "homeTeam_logo",
            "homeTeam_darkLogo",
            "periodDescriptor_number",
            "periodDescriptor_periodType",
            "periodDescriptor_otPeriods",
            "periodDescriptor_maxRegulationPeriods",
            "gameOutcome_lastPeriodType",
            "winningGoalie_playerId",
            "winningGoalScorer_playerId",
            "clock_timeRemaining",
            "clock_secondsRemaining",
            "clock_running",
            "clock_inIntermission",
            "gameCenterLink",
        ]
        for column in expected_columns:
            if column not in df.columns:
                df[column] = None

        return df

    def get_game_team_rosters(self, schedule_games: List[Dict], rosters_df: pd.DataFrame) -> pd.DataFrame:
        """Expand team rosters into one row per game/team/player."""
        expected_columns = [
            "scheduleDate",
            "gameId",
            "teamSide",
            "teamAbbreviation",
            "opponentAbbreviation",
            "positionGroup",
            "playerId",
            "headshot",
            "firstName",
            "lastName",
            "fullName",
            "sweaterNumber",
            "positionCode",
            "shootsCatches",
            "heightInInches",
            "weightInPounds",
            "heightInCentimeters",
            "weightInKilograms",
            "birthDate",
            "birthCity",
            "birthCountry",
            "birthStateProvince",
        ]

        if rosters_df is None or rosters_df.empty or not schedule_games:
            return pd.DataFrame(columns=expected_columns)

        rows = []
        for game in schedule_games:
            schedule_date = game.get("scheduleDate") or game.get("gameDate")
            game_id = game.get("id")
            away_abbrev = game.get("awayTeam", {}).get("abbrev")
            home_abbrev = game.get("homeTeam", {}).get("abbrev")

            for team_side, team_abbrev, opponent_abbrev in (
                ("away", away_abbrev, home_abbrev),
                ("home", home_abbrev, away_abbrev),
            ):
                if not team_abbrev:
                    continue

                team_roster = rosters_df[rosters_df["teamAbbreviation"] == team_abbrev]
                for player in team_roster.to_dict(orient="records"):
                    rows.append({
                        "scheduleDate": schedule_date,
                        "gameId": game_id,
                        "teamSide": team_side,
                        "teamAbbreviation": team_abbrev,
                        "opponentAbbreviation": opponent_abbrev,
                        **player,
                    })

        df = pd.DataFrame(rows)
        for column in expected_columns:
            if column not in df.columns:
                df[column] = None

        return df[expected_columns]

    async def scrape_schedule_now_game_rosters(self) -> Dict[str, pd.DataFrame]:
        """Fetch /schedule/now games and current rosters for participating teams."""
        schedule_games = self.get_schedule_now_games()
        games_df = self.get_schedule_now_games_dataframe()
        teams = sorted({
            team_abbrev
            for game in schedule_games
            for team_abbrev in (
                game.get("awayTeam", {}).get("abbrev"),
                game.get("homeTeam", {}).get("abbrev"),
            )
            if team_abbrev
        })

        if teams:
            team_rosters = await self.scrape_all_rosters(team_codes=teams)
            game_rosters = self.get_game_team_rosters(schedule_games, team_rosters)
        else:
            team_rosters = pd.DataFrame()
            game_rosters = self.get_game_team_rosters([], team_rosters)

        return {
            "games": games_df,
            "teams": teams,
            "team_rosters": team_rosters,
            "game_rosters": game_rosters,
        }

    async def scrape_recent_schedule_game_rosters(
        self,
        days: int = 2,
        end_date: Optional[str] = None,
        lookahead_days: int = 0,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch games and current rosters for teams playing in a recent date window."""
        dates = self.get_schedule_window_dates(
            days=days,
            end_date=end_date,
            lookahead_days=lookahead_days,
        )
        schedule_games = self.get_games_for_dates(dates)
        games_df = (
            self._ensure_games_staging_columns(self._games_to_dataframe(schedule_games))
            if schedule_games else pd.DataFrame()
        )
        teams = sorted({
            team_abbrev
            for game in schedule_games
            for team_abbrev in (
                game.get("awayTeam", {}).get("abbrev"),
                game.get("homeTeam", {}).get("abbrev"),
            )
            if team_abbrev
        })
        known_teams = [team for team in teams if team in self.active_team_codes]
        unknown_teams = [team for team in teams if team not in self.active_team_codes]
        if unknown_teams:
            self.logger.warning("Skipping unknown schedule team code(s): %s", unknown_teams)

        if known_teams:
            team_rosters = await self.scrape_all_rosters(team_codes=known_teams)
            game_rosters = self.get_game_team_rosters(schedule_games, team_rosters)
        else:
            team_rosters = pd.DataFrame()
            game_rosters = self.get_game_team_rosters([], team_rosters)

        return {
            "games": games_df,
            "teams": known_teams,
            "dates": dates,
            "team_rosters": team_rosters,
            "game_rosters": game_rosters,
        }

    def get_todays_teams(self, date: Optional[str] = None) -> List[str]:
        """
        Get list of team tricodes playing today (or on a specific date).

        Args:
            date: Optional date string in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            List of unique team tricodes playing on that date
        """
        games = self.get_todays_games(date)
        teams = set()

        for game in games:
            away = game.get('awayTeam', {}).get('abbrev')
            home = game.get('homeTeam', {}).get('abbrev')
            if away:
                teams.add(away)
            if home:
                teams.add(home)

        return list(teams)

    def get_todays_schedule(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get a summary of today's games with start times and scores.

        Args:
            date: Optional date string in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            DataFrame with game info: teams, start time, venue, scores, game state
        """
        games = self.get_todays_games(date)
        if not games:
            return pd.DataFrame()

        rows = []
        for game in games:
            away_team = game.get('awayTeam', {})
            home_team = game.get('homeTeam', {})

            rows.append({
                'gameId': game.get('id'),
                'startTimeUTC': game.get('startTimeUTC'),
                'gameState': game.get('gameState'),
                'awayTeam': away_team.get('abbrev'),
                'awayScore': away_team.get('score'),
                'homeTeam': home_team.get('abbrev'),
                'homeScore': home_team.get('score'),
                'venue': game.get('venue', {}).get('default', ''),
                'tvBroadcasts': ', '.join([b.get('network', '') for b in game.get('tvBroadcasts', [])]),
            })

        return pd.DataFrame(rows)

    def get_todays_games_dataframe(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get today's games as a DataFrame in the same format as scrape_all_games_to_dataframe.
        Compatible for SQL upsert operations.

        Args:
            date: Optional date string in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            DataFrame with game data, flattened for SQL compatibility
        """
        games = self.get_todays_games(date)
        if not games:
            return pd.DataFrame()

        df = pd.json_normalize(games, sep='_')

        # Convert tvBroadcasts list to comma-separated string of networks
        if 'tvBroadcasts' in df.columns:
            df['tvBroadcasts'] = df['tvBroadcasts'].apply(
                lambda x: ', '.join([b.get('network', '') for b in x]) if isinstance(x, list) else ''
            )

        return self._ensure_games_staging_columns(df)

    async def scrape_todays_team_stats(self, date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Scrape current season stats only for teams playing today.

        Args:
            date: Optional date string in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            Dict with 'skaters', 'goalies', 'schedule' DataFrames
        """
        teams = self.get_todays_teams(date)
        if not teams:
            self.logger.info("No games scheduled for today")
            return {'skaters': pd.DataFrame(), 'goalies': pd.DataFrame(), 'schedule': pd.DataFrame()}

        self.logger.info(f"Teams playing today: {', '.join(teams)}")

        # Get stats for just those teams
        stats = await self._scrape_current_season_async(team_codes=teams)

        # Add today's schedule
        stats['schedule'] = self.get_todays_schedule(date)

        return stats


async def main():
    scraper = NHLScraper()
    
    # Scrape current season data
    current_data = await scraper.scrape_current_season()
    
    # Save to CSV files with current season in filename
    season = scraper.get_current_season()
    for key, df in current_data.items():
        filename = f"nhl_{key}_{season}.csv"
        df.to_csv(filename, index=False)
        
        # Print summary statistics
        print(f"\n{key.upper()} Summary:")
        print(f"Total records: {len(df)}")
        
        if key in ['skaters', 'goalies']:
            print("\nRecords by game type:")
            print(df.groupby('gameType').size())
            
        if key == 'skaters':
            for game_type in df['gameType'].unique():
                game_type_name = 'Playoffs' if game_type == 3 else 'Regular Season'
                print(f"\nTop 5 scorers - {game_type_name}:")
                game_type_df = df[df['gameType'] == game_type]
                top_scorers = game_type_df.nlargest(5, 'points')[
                    ['firstName', 'lastName', 'team', 'goals', 'assists', 'points']]
                print(top_scorers)
    
    # Scrape all team rosters
    print("\n" + "="*50)
    print("ROSTER SCRAPING")
    print("="*50)
    
    rosters_df = await scraper.scrape_all_rosters()
    rosters_df.to_csv(f"nhl_rosters_{season}.csv", index=False)
    
    print(f"\nTotal players across all rosters: {len(rosters_df)}")
    print("\nPlayers by position:")
    print(rosters_df.groupby('position').size())
    print("\nPlayers per team:")
    print(rosters_df.groupby('teamAbbreviation').size())

if __name__ == "__main__":
    asyncio.run(main())
