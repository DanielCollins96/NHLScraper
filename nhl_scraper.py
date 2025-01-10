import requests
import pandas as pd
from time import sleep
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
from datetime import datetime

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


    def scrape_current_season(self, team_codes: Optional[Union[str, List[str]]] = None) -> Dict[str, pd.DataFrame]:
        """Scrape current season data for all game types
        return Dict has keys 'teams', 'skaters', 'goalies' with corresponding DataFrames
        """
        all_skaters = []
        all_goalies = []
        processed_teams = []
        
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
        
        for team in teams_to_scrape:
            team_has_data = False
            
            for game_type in self.game_types:
                try:
                    skaters_df, goalies_df = self.get_team_current_stats(team, game_type)
                    
                    if not skaters_df.empty or not goalies_df.empty:
                        team_has_data = True
                        
                        # Add team information
                        if not skaters_df.empty:
                            skaters_df['team'] = team
                            all_skaters.append(skaters_df)
                        
                        if not goalies_df.empty:
                            goalies_df['team'] = team
                            all_goalies.append(goalies_df)
                    
                    sleep(.1)  # Rate limiting
                except Exception as e:
                    self.logger.error(f"Error processing team {team} game type {game_type}: {e}")
                    continue
            
            if team_has_data:
                processed_teams.append({'triCode': team})
        
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
    
    def scrape_team_gametypes(self, tricode: str):
        """Scrape all gametypes (Seasons Reg/PO) for a team(tricode)"""
        url = f"{self.web_api_url}/club-stats-season/{tricode}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    def scrape_all_seasons_by_gametype(self, active_only: bool = True):
        """
        Scrape all team data, including gametypes and seasons.
        Optionally restrict to active teams only.
        """
        all_data = []
        teams = self.get_all_teams()['triCode'].tolist()
        if active_only:
            teams = [team for team in teams if team in self.active_team_codes]

        for tricode in teams[:5]:

            try:
                gametypes_data = self.scrape_team_gametypes(tricode)
                for entry in gametypes_data:
                    season = entry['season']
                    for game_type in entry['gameTypes']: 
                        url = f"{self.web_api_url}/club-stats/{tricode}/{season}/{game_type}"
                        print(url)
                        all_data.append(url)

                print(f"Scraped data for team {tricode}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching data for team {tricode}: {e}")
                continue

        # Convert to a pandas DataFrame
        # return pd.DataFrame(all_data)
        return all_data
    
    def scrape_player_season_totals(self, player_id: str) -> pd.DataFrame:
        url = f"{self.web_api_url}/player/{player_id}/landing"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data 
        player_df = pd.DataFrame(data)
        return player_df
    
    def _data_to_skaters_and_goalies_df(self, data, game_type, season, tricode) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Processes a Dict[str, DF] skater/player dict (output from `scrape_current_season` etc.) into skaters and goalies DataFrames
        """
        skaters_df = pd.DataFrame(data.get('skaters', []))
        if not skaters_df.empty:
            skaters_df['firstName'] = skaters_df['firstName'].apply(lambda x: x.get('default', ''))
            skaters_df['lastName'] = skaters_df['lastName'].apply(lambda x: x.get('default', ''))
            skaters_df['gameType'] = game_type
            skaters_df['season'] = season
            skaters_df['team'] = tricode
        
        # Process goalies
        goalies_df = pd.DataFrame(data.get('goalies', []))
        if not goalies_df.empty:
            goalies_df['firstName'] = goalies_df['firstName'].apply(lambda x: x.get('default', ''))
            goalies_df['lastName'] = goalies_df['lastName'].apply(lambda x: x.get('default', ''))
            goalies_df['gameType'] = game_type
            goalies_df['season'] = season
            goalies_df['team'] = tricode
        
        return skaters_df, goalies_df
        

def main():
    scraper = NHLScraper()
    
    # Scrape current season data
    current_data = scraper.scrape_current_season()
    
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
    main()
