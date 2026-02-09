import requests
import pandas as pd
from time import sleep
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
from datetime import datetime
import aiohttp
import asyncio
from sqlalchemy import text

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
        """Calculate the current NHL season string (October start date)"""
        current_date = datetime.now()
        current_year = current_date.year
        if current_date.month >= 10:
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
                            all_awards.append(awards_df)

                current_batch = i // batch_size + 1
                total_batches = (len(urls) + batch_size - 1) // batch_size
                self.logger.info(f"Processed batch {current_batch}/{total_batches} ({len(batch_urls)} players)")

            with engine.begin() as conn:
                if all_players:
                    try:
                        pd.concat(all_players, ignore_index=True).to_sql('player', conn, if_exists='replace', index=False, schema='staging1')
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.player: {e}")
                        raise
                if all_skater_seasons:
                    try:
                        pd.concat(all_skater_seasons, ignore_index=True).to_sql('season_skater', conn, if_exists='replace', index=False, schema='staging1')
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.season_skater: {e}")
                        raise
                if all_goalie_seasons:
                    try:
                        pd.concat(all_goalie_seasons, ignore_index=True).to_sql('season_goalie', conn, if_exists='replace', index=False, schema='staging1')
                    except Exception as e:
                        self.logger.error(f"Failed to insert into staging1.season_goalie: {e}")
                        raise
                if all_awards:
                    try:
                        pd.concat(all_awards, ignore_index=True).to_sql('award', conn, if_exists='replace', index=False, schema='staging1')
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

    def scrape_all_games_to_dataframe(self, season: str = "now") -> pd.DataFrame:
        """
        Scrape all games and return as a DataFrame.

        Args:
            season: Season string (e.g., '20232024') or 'now' for current season

        Returns:
            DataFrame with game data, flattened for SQL compatibility
        """
        games = self.scrape_all_games_team_method(season)
        if not games:
            return pd.DataFrame()

        df = pd.json_normalize(games, sep='_')

        # Convert tvBroadcasts list to comma-separated string of networks
        if 'tvBroadcasts' in df.columns:
            df['tvBroadcasts'] = df['tvBroadcasts'].apply(
                lambda x: ', '.join([b.get('network', '') for b in x]) if isinstance(x, list) else ''
            )

        return df

    def get_todays_games(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get all games scheduled for today (or a specific date).

        Args:
            date: Optional date string in 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            List of game dictionaries for the specified date
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

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

        return df

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