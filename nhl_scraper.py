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
    
    def get_all_drafts(self) -> pd.DataFrame:
        """"""
        url = f"{self.base_url}/draft"
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

    async def scrape_team_gametypes(self, tricode: Optional[str] = None) -> pd.DataFrame:
        """Scrape all gametypes (Seasons Reg/PO) for a team(tricode) or all active teams if no tricode is provided."""
        if tricode is None:
            tricodes = self.active_team_codes
        else:
            tricodes = [tricode]

        urls = [f"{self.web_api_url}/club-stats-season/{code}" for code in tricodes]

        # Fetch all data asynchronously
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
        # Gather URLs first
        all_urls = []
        teams = self.get_all_teams()['triCode'].tolist()
        if active_only:
            teams = [team for team in teams if team in self.active_team_codes]

        for tricode in teams:
            try:
                gametypes_data = await self.scrape_team_gametypes(tricode)
                for entry in gametypes_data.to_dict(orient='records'):
                    season = entry['season']
                    for game_type in entry['gameTypes']: 
                        url = f"{self.web_api_url}/club-stats/{tricode}/{season}/{game_type}"
                        # Create a dictionary with metadata
                        url_data = {
                            'url': url,
                            'tricode': tricode,
                            'season': season,
                            'gameType': game_type
                        }
                        all_urls.append(url_data)
                    self.logger.info(f"Gathered URLs for team {tricode}")
            except Exception as e:
                self.logger.error(f"Error gathering URLs for team {tricode}: {e}")
                continue

        # Extract just the URLs for the API call
        urls = [item['url'] for item in all_urls]
        
        # Fetch all data asynchronously
        responses = await self._fetch_all_data(urls)
        
        # Combine responses with metadata
        processed_responses = []
        for response, metadata in zip(responses, all_urls):
            if response is not None:
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
        
    async def process_all_teams(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes all team data from `scraper.scrape_all_seasons_by_gametype()` 
        and merges all skater and goalie DataFrames.
        
        Args:
            scraper: Instance of the scraper class.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Combined Skaters and Goalies DataFrames.
        """
        # Step 1: Fetch all team data
        data = await self.scrape_all_seasons_by_gametype()

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
        # Create URLs for all players
        try:
            with engine.connect() as connection:
                connection.execute(text("TRUNCATE TABLE newapi.player, newapi.season, newapi.award"))
                connection.commit()

            urls = [f"{self.web_api_url}/player/{player_id}/landing" for player_id in player_ids]
            # Process in batches
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_ids = player_ids[i:i + batch_size]
                
                # Fetch batch of data
                responses = await self._fetch_all_data(batch_urls)
                
                # Process responses
                all_players = []
                all_seasons = []
                all_awards = []
                
                for response, player_id in zip(responses, batch_ids):
                    if response:
                            player_df = pd.json_normalize(response)
                            seasons_df = self._process_season_totals(player_df, player_id)
                            awards_df = (self._process_awards(player_df, player_id) 
                                    if 'awards' in player_df.columns 
                                    else pd.DataFrame())

                            cols_to_drop = ['badges', 'last5Games', 'seasonTotals', 
                                        'currentTeamRoster', 'awards', 'shopLink', 
                                        'twitterLink', 'watchLink']
                            existing_cols = [col for col in cols_to_drop if col in player_df.columns]
                            player_df.drop(columns=existing_cols, inplace=True)
                            
                            all_players.append(player_df)
                            all_seasons.append(seasons_df)
                            all_awards.append(awards_df)

                with engine.connect() as conn:
                    if all_players:
                        pd.concat(all_players).to_sql('player', conn, if_exists='append', index=False, schema='newapi')
                    if all_seasons:
                        pd.concat(all_seasons).to_sql('season', conn, if_exists='append', index=False, schema='newapi')
                    if all_awards:
                        pd.concat(all_awards).to_sql('award', conn, if_exists='append', index=False, schema='newapi')
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error writing batch to database: {e}")
            raise
        self.logger.info(f"Processed batch {i//batch_size + 1}, {len(batch_urls)} players")
            

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
            if game_type:
                goalies_df['gameType'] = game_type
            if season:
                goalies_df['season'] = season
            if tricode:
                goalies_df['team'] = tricode
        
        return skaters_df, goalies_df
    
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
    
    async def _fetch_all_data(self, urls: List[str], batch_size: int = 10) -> List[dict]:
        """Fetch data from multiple URLs concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_data(session, url) for url in urls]
            return await asyncio.gather(*tasks)
        

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

if __name__ == "__main__":
    asyncio.run(main())